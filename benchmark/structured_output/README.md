# Warm structured-output profiling

[Measured H100 results](H100_RESULTS.md): model forward dominated the tested
Qwen3-4B workloads; CPU mask fill already overlapped GPU decode, and padded
vocabulary entries prevented the unrestricted-mask fast path from firing.

This harness compares unconstrained, simple-schema, and supplied-schema requests
at batch sizes 1 and 4. It warms each case, records unprofiled end-to-end wall times,
then captures a separate CPU/GPU trace. Use a dedicated, otherwise idle SGLang
server with `--grammar-backend xgrammar` and the same model, TP configuration,
sampling options, and graph settings for both revisions.

```bash
python benchmark/structured_output/profile_masks.py run \
  --url http://127.0.0.1:30000 \
  --schema /path/to/production-schema.json \
  --prompt /path/to/prompt.txt \
  --output /tmp/structured-output-manifests \
  --trace-dir /tmp/structured-output-traces

python benchmark/structured_output/profile_masks.py summarize \
  /path/to/downloaded/trace.trace.json.gz
```

The trace directory is on the **server**, while manifests are on the **client**.
Manifests retain the actual `sampling_params.json_schema`, server information,
responses, completion token counts, and finish reasons. The prompt is the same
across cases; make it request enough text for meaningful decode work. Equal token
limits do not ensure equal output lengths: compare completion token counts and
decode-only iterations, and check for truncation and valid schema-conforming JSON.
Requested concurrency does not guarantee every decode iteration has that batch
size; inspect the trace. The first run warms compilation and execution, but
additional warmups may be needed for a particular model or runtime.

The added spans are:

| Span | Meaning |
| --- | --- |
| `grammar.allocate` | Host buffer allocation and initialization |
| `grammar.fill` | Active matcher rows, including reasoning-wrapper decisions |
| `grammar.transfer` | Host-side mask transfer call |
| `grammar.apply` | Host-side mask application call |

Spans use the existing profiler helper, which emits PyTorch ranges when profiling
and optionally NVTX with `SGLANG_ENABLE_NVTX_OPERATIONS=1` and the `nvtx` package.
Unrestricted steps have fill ranges but no transfer/apply ranges. Requests whose
grammars are all inactive have no grammar ranges. Older builds do not emit these
names, so absence in an existing trace is inconclusive.

The summary separates **CPU span** durations from profiler-correlated **GPU
intervals**, including forward spans. Inspect copy/kernel lanes in Perfetto or
Nsight Systems for details; do not add overlapping CPU and GPU durations or nested
spans. `aten::full` alone does not prove grammar
use. A grammar fill range or a request manifest carrying an active constraint is
stronger evidence; a forward-only capture can miss sampling and all grammar work.

## Initial investigation

Based on mainline `525f14040dd77da760b713848a803fdd4d32c2b7`:

- `XGrammarGrammar.fill_vocab_mask` discarded the matcher's `need_apply` result.
  The patch preserves it through batch filling and reasoning wrappers, and skips
  transfer/application only when every active row explicitly returns `False`.
  Legacy backends returning `None` remain conservative. All rows are filled even
  after one restricted row is found.
- Batches with only finished, terminated, or absent grammars now skip allocation.
- Active XGrammar batches still allocate and initialize CPU masks. Pinned CPU/GPU
  buffer reuse, forward overlap, TP-rank consolidation, and jump-forward changes
  are follow-up experiments, not implemented or claimed as measured improvements.
  Reuse must protect pinned host storage until async H2D completes and device
  storage until the last consumer completes, including overlap and speculative
  paths. A stable tensor address alone is not sufficient.

Initially, no saved `.pt.trace.json*`, `.trace.json*`, `.nsys-rep`, or `.pftrace` files were
found in the accessible workspace and `/tmp` search during this investigation.
The existing Dynamo `examples/backends/sglang/test_sglang_profile.py` uses ordinary
`/v1/completions` requests without a schema; it is not a structured-output profile.
GPU access subsequently became available, and the comparison above collected 16
forward-pass traces on an H100 with structured-output requests explicitly
identified in manifests and grammar spans.

Validation: 113 tests passed across the sampling-batch, base-grammar, and
reasoner-grammar suites, including real XGrammar mixed-row masking and the
transition from restricted output to an unrestricted batch on CPU and CUDA.
Two client tests cover plain-text/empty profiling endpoint responses and separate
CPU/GPU annotation summaries in plain/gzip traces.

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/sampling/test_sampling_batch_info.py \
  test/registered/unit/constrained/test_base_grammar_backend.py \
  test/registered/unit/constrained/test_reasoner_grammar_backend.py
python benchmark/structured_output/test_profile_masks.py
```

Captures start on the next forward pass to avoid recording idle scheduler spins.
Stack collection is disabled by default; enable it with `--with-stack` when
needed. `--cases` can select individual cases or the optional `structural` case,
which wraps the supplied schema inside `<report>...</report>` and allows free
text outside the tags. Use `example-tagged-prompt.txt` with that case; the supplied
example prompts already contain Qwen ChatML framing.

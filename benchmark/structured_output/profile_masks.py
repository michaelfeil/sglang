"""Capture warmed structured-output comparisons on a running SGLang server.

Uses only the standard library. Trace paths are on the SERVER filesystem;
manifests and unprofiled wall times are written on the CLIENT filesystem.
"""

import argparse
import gzip
import json
import math
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def request(base_url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        payload = response.read()
        # Profiling control endpoints return plain text (or an empty body).
        if path in {"/start_profile", "/stop_profile"}:
            return payload.decode() if payload else None
        return json.loads(payload)


def run(args):
    schema = json.loads(args.schema.read_text())
    prompt = args.prompt.read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    server_info = request(args.url, "/get_server_info")
    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<report>"],
            "tags": [
                {
                    "begin": "<report>",
                    "content": {"type": "json_schema", "json_schema": schema},
                    "end": "</report>",
                }
            ],
        },
    }
    for name, constraint in (
        ("unconstrained", None),
        ("simple", SIMPLE_SCHEMA),
        ("complex", schema),
        ("structural", structural_tag),
    ):
        if name not in args.cases:
            continue
        for batch_size in args.batches:
            case = f"{name}-b{batch_size}"
            params = {"temperature": 0, "max_new_tokens": args.max_tokens}
            if constraint is not None:
                key = "structural_tag" if name == "structural" else "json_schema"
                params[key] = json.dumps(constraint)
            payload = {"text": [prompt] * batch_size, "sampling_params": params}
            # Warm the grammar cache and model before either timing or tracing.
            request(args.url, "/generate", payload)
            measurements = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = request(args.url, "/generate", payload)
                measurements.append(
                    {"wall_seconds": time.perf_counter() - start, "responses": result}
                )
            manifest = {
                "case": case,
                "server_info": server_info,
                "request": payload,
                "measurements": measurements,
                "server_trace_dir": f"{args.trace_dir.rstrip('/')}/{case}",
                "note": "Requested batch size; verify actual decode batch sizes in trace."
                " Wall times include prefill, sampling, transport, and serialization.",
            }
            manifest_path = args.output / f"{case}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            request(
                args.url,
                "/start_profile",
                {
                    "output_dir": manifest["server_trace_dir"],
                    "activities": ["CPU", "GPU"],
                    # Start on the next forward, avoiding idle scheduler spins.
                    "start_step": 1,
                    "with_stack": args.with_stack,
                    "record_shapes": True,
                    "profile_prefix": case,
                },
            )
            try:
                manifest["profiled_responses"] = request(args.url, "/generate", payload)
            finally:
                request(args.url, "/stop_profile", {})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"{case}: saved {manifest_path}", flush=True)


def summarize(args):
    for path in args.traces:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as source:
            trace = json.load(source)
        events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
        ranges = defaultdict(list)
        gpu_ranges = defaultdict(list)
        for event in events:
            name = event.get("name", "")
            if event.get("ph") != "X" or not name.startswith(("grammar.", "step[")):
                continue
            if event.get("cat") == "user_annotation":
                ranges[name].append(event["dur"])
            elif event.get("cat") == "gpu_user_annotation":
                gpu_ranges[name].append(event["dur"])
        report = {
            "trace": str(path),
            "structured_output_evidence": (
                "grammar fill observed" if "grammar.fill" in ranges else "inconclusive"
            ),
            "cpu_ranges_us": {},
            "gpu_ranges_us": {},
            "note": "Host ranges measure CPU work or GPU enqueue time, not GPU duration."
            " GPU ranges are profiler-correlated device intervals; do not add nested spans."
            " Missing ranges in an older or partial trace do not prove absence of constraints.",
        }
        for key, group in (("cpu_ranges_us", ranges), ("gpu_ranges_us", gpu_ranges)):
            for name, values in sorted(group.items()):
                values.sort()
                report[key][name] = {
                    "count": len(values),
                    "median": statistics.median(values),
                    "p95": values[math.ceil(0.95 * len(values)) - 1],
                    "max": values[-1],
                }
        print(json.dumps(report, indent=2))


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("run")
    capture.add_argument("--url", default="http://127.0.0.1:30000")
    capture.add_argument("--schema", type=Path, required=True)
    capture.add_argument("--prompt", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--trace-dir", required=True, help="Absolute server-side path")
    capture.add_argument("--batches", type=positive_int, nargs="+", default=[1, 4])
    capture.add_argument("--max-tokens", type=positive_int, default=256)
    capture.add_argument("--repeats", type=positive_int, default=5)
    capture.add_argument(
        "--cases",
        nargs="+",
        choices=["unconstrained", "simple", "complex", "structural"],
        default=["unconstrained", "simple", "complex"],
    )
    capture.add_argument("--with-stack", action="store_true")
    capture.set_defaults(func=run)
    summary = commands.add_parser("summarize")
    summary.add_argument("traces", type=Path, nargs="+")
    summary.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

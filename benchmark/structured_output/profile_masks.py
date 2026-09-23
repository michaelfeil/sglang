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
        return json.load(response)


def run(args):
    schema = json.loads(args.schema.read_text())
    prompt = args.prompt.read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    server_info = request(args.url, "/get_server_info")
    for name, constraint in (
        ("unconstrained", None),
        ("simple", SIMPLE_SCHEMA),
        ("complex", schema),
    ):
        for batch_size in args.batches:
            case = f"{name}-b{batch_size}"
            params = {"temperature": 0, "max_new_tokens": args.max_tokens}
            if constraint is not None:
                params["json_schema"] = json.dumps(constraint)
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
                    "with_stack": True,
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
        for event in events:
            if event.get("ph") == "X" and event.get("name", "").startswith("grammar."):
                ranges[event["name"]].append(event["dur"])
        report = {
            "trace": str(path),
            "structured_output_evidence": (
                "grammar fill observed" if "grammar.fill" in ranges else "inconclusive"
            ),
            "cpu_ranges_us": {},
            "note": "Host ranges measure CPU work or GPU enqueue time, not GPU duration."
            " Inspect CUDA kernels, copies, forward spans, and NCCL lanes for device time."
            " Missing ranges in an older or partial trace do not prove absence of constraints.",
        }
        for name, values in sorted(ranges.items()):
            values.sort()
            report["cpu_ranges_us"][name] = {
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
    capture.set_defaults(func=run)
    summary = commands.add_parser("summarize")
    summary.add_argument("traces", type=Path, nargs="+")
    summary.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

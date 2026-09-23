"""Validate capture manifests and compare baseline/patched CPU and GPU traces.

Run with the evidence directory containing baseline[-traces]/patched[-traces].
Requires jsonschema for validation; does not require a GPU.
"""

import argparse
import bisect
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import jsonschema


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "min": values[0],
        "max": values[-1],
    }


def analyze_case(manifest, trace_path):
    data = json.loads(manifest.read_text())
    measurements = data["measurements"]
    responses = [r for m in measurements for r in m["responses"]]
    params = data["request"]["sampling_params"]
    schema = params.get("json_schema")
    tagged = "structural_tag" in params
    if tagged:
        schema = json.loads(params["structural_tag"])["format"]["tags"][0]["content"][
            "json_schema"
        ]
    elif schema:
        schema = json.loads(schema)
    validation = []
    for response in responses + data.get("profiled_responses", []):
        if schema:
            text = response["text"]
            if tagged:
                matches = re.findall(r"<report>(.*?)</report>", text, flags=re.DOTALL)
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one report in {manifest}")
                text = matches[0]
            jsonschema.validate(json.loads(text), schema)
            validation.append(True)

    with gzip.open(trace_path, "rt") as source:
        events = json.load(source)["traceEvents"]
    host, device = defaultdict(list), defaultdict(list)
    fills, forwards, allocations, full_ops = [], [], [], []
    for event in events:
        if event.get("ph") != "X":
            continue
        name, category = event["name"], event.get("cat")
        if category == "user_annotation":
            if name.startswith(("grammar.", "step[")):
                host[name].append(event["dur"])
            if name == "grammar.fill":
                fills.append(event)
            if name == "grammar.allocate":
                allocations.append(event)
        elif category == "gpu_user_annotation":
            if name.startswith(("grammar.", "step[")):
                device[name].append(event["dur"])
            if name.startswith("step[DECODE"):
                forwards.append(event)
        elif category == "cpu_op" and name == "aten::full":
            full_ops.append(event)

    def contained_count(inner, outer):
        outer = sorted(outer, key=lambda e: e["ts"])
        starts = [e["ts"] for e in outer]
        contained = []
        for event in inner:
            idx = bisect.bisect_right(starts, event["ts"]) - 1
            if (
                idx >= 0
                and event["ts"] + event["dur"] <= outer[idx]["ts"] + outer[idx]["dur"]
            ):
                contained.append(event)
        return contained

    return {
        "manifest": str(manifest),
        "trace": str(trace_path),
        "wall_seconds": stats([m["wall_seconds"] for m in measurements]),
        "completion_tokens_per_request": sorted(
            {r["meta_info"]["completion_tokens"] for r in responses}
        ),
        "finish_reasons": sorted(
            {r["meta_info"]["finish_reason"]["type"] for r in responses}
        ),
        "validated_responses_including_profile": len(validation),
        "cpu_us": {k: stats(v) for k, v in sorted(host.items())},
        "gpu_us": {k: stats(v) for k, v in sorted(device.items())},
        "allocation_aten_full_cpu_us": stats(
            [e["dur"] for e in contained_count(full_ops, allocations)]
        ),
        "matcher_fills_fully_inside_gpu_decode": len(contained_count(fills, forwards)),
        "matcher_fill_count": len(fills),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    report = {"variants": {}, "comparisons": {}}
    texts = {}
    for variant in ("baseline", "patched"):
        report["variants"][variant] = {}
        for manifest in sorted((args.evidence / variant).glob("*.json")):
            case = manifest.stem
            traces = list((args.evidence / f"{variant}-traces" / case).glob("*.gz"))
            if len(traces) != 1:
                raise ValueError(
                    f"Expected one trace for {variant}/{case}, got {traces}"
                )
            report["variants"][variant][case] = analyze_case(manifest, traces[0])
            data = json.loads(manifest.read_text())
            texts[variant, case] = [
                [r["text"] for r in m["responses"]] for m in data["measurements"]
            ]
    for case, baseline in report["variants"]["baseline"].items():
        patched = report["variants"]["patched"][case]
        report["comparisons"][case] = {
            "identical_output_texts": texts["baseline", case] == texts["patched", case],
            "patched_wall_change_percent": 100
            * (
                patched["wall_seconds"]["median"] / baseline["wall_seconds"]["median"]
                - 1
            ),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Render a short CPU/GPU timeline from a captured trace (requires matplotlib)."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as source:
        events = json.load(source)["traceEvents"]
    forwards = sorted(
        (
            e
            for e in events
            if e.get("cat") == "gpu_user_annotation"
            and e.get("name", "").startswith("step[DECODE")
        ),
        key=lambda e: e["ts"],
    )
    start = forwards[args.step]["ts"] - 300
    end = forwards[args.step + 1]["ts"] + forwards[args.step + 1]["dur"] + 400
    lanes = [
        ("CPU forward enqueue", "user_annotation", "step[DECODE", "#64748b"),
        ("CPU mask allocation", "user_annotation", "grammar.allocate", "#d97706"),
        ("CPU matcher fill", "user_annotation", "grammar.fill", "#c2410c"),
        ("CPU copy enqueue", "user_annotation", "grammar.transfer", "#0369a1"),
        ("CPU apply enqueue", "user_annotation", "grammar.apply", "#7c3aed"),
        ("GPU forward", "gpu_user_annotation", "step[DECODE", "#059669"),
        ("GPU mask copy", "gpu_user_annotation", "grammar.transfer", "#0369a1"),
        ("GPU mask application", "gpu_user_annotation", "grammar.apply", "#7c3aed"),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    for row, (label, category, prefix, color) in enumerate(lanes):
        spans = [
            ((e["ts"] - start) / 1000, e["dur"] / 1000)
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == category
            and e["name"].startswith(prefix)
            and e["ts"] < end
            and e["ts"] + e["dur"] > start
        ]
        ax.broken_barh(spans, (row - 0.3, 0.6), facecolors=color)
        if row >= 6:
            # Two-microsecond GPU operations are subpixel at this scale.
            ax.vlines(
                [s[0] for s in spans],
                row - 0.25,
                row + 0.25,
                color=color,
                linewidth=0.8,
            )
    ax.set_yticks(range(len(lanes)), [lane[0] for lane in lanes])
    ax.invert_yaxis()
    ax.set_xlim(0, (end - start) / 1000)
    ax.set_xlabel(
        "Elapsed time in this trace window (ms)\nGPU mask ticks mark operations too short to resolve at this scale."
    )
    ax.set_title(
        "H100 · Qwen3-4B · complex schema · batch 1\nCPU matcher work overlaps GPU forward"
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(args.output)
    if args.output.suffix == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines())
            + "\n"
        )


if __name__ == "__main__":
    main()

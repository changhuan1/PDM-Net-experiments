from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate test_metrics.json files into CSV summaries.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/summary.csv"))
    return parser.parse_args()


def parse_run_name(run_name: str) -> dict[str, str]:
    parts = run_name.split("_")
    shot = next((p for p in parts if p.endswith("shot") or p == "full"), "")
    seed = next((p for p in parts if p.startswith("seed")), "")
    model = run_name
    if shot:
        model = run_name.split(f"_{shot}")[0]
    return {"model": model, "shot": shot, "seed": seed.replace("seed", "")}


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in args.outputs.glob("*/test_metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"run": metrics_path.parent.name, **parse_run_name(metrics_path.parent.name), **metrics}
        rows.append(row)

    if not rows:
        raise SystemExit(f"No test_metrics.json files found under {args.outputs}")

    df = pd.DataFrame(rows).sort_values(["model", "shot", "seed"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    grouped = (
        df.groupby(["model", "shot"])[["accuracy", "macro_f1", "weighted_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.to_csv(args.out.with_name(args.out.stem + "_grouped.csv"), index=False)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_name(args.out.stem + '_grouped.csv')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.fewshot_split import make_fewshot_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create few-shot CSV splits for facial affect datasets.")
    parser.add_argument("--source", required=True, type=Path, help="Dataset root under data/raw/<dataset_name>.")
    parser.add_argument("--out", required=True, type=Path, help="Output split directory.")
    parser.add_argument("--shots", nargs="+", default=["5", "10", "20", "50", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_fewshot_splits(
        source=args.source,
        out=args.out,
        shots=args.shots,
        seeds=args.seeds,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(f"Wrote few-shot splits to {args.out}")


if __name__ == "__main__":
    main()

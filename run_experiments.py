#!/usr/bin/env python
"""
CLI entry point.

Dry run (no API calls, no cost, validates the whole pipeline wiring):
    python run_experiments.py --mock

Real run, full grid, sequential (requires OPENROUTER_API_KEY):
    python run_experiments.py

Real run, a SHARD of the grid (for running several shards in parallel
background processes -- see PARALLEL_RUN.md):
    python run_experiments.py --only-framework negotiation_tool --only-metric cost --only-model-tier small --out results/shard_neg_cost_small.json

Merge shard result files back into one combined results file + table:
    python run_experiments.py --merge results/shard_*.json --out results/results.json

Optional: run a short pilot (fewer steps) before committing to the full
200-step x 25-config grid, to sanity-check costs/behaviour and catch
malformed-output issues early.
    python run_experiments.py --steps 20
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src import config as cfg
from src.experiment_runner import build_full_grid, filter_grid, merge_result_files, run_full_grid
from src.results_table import format_all_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use a free deterministic mock LLM to validate the pipeline (no API calls, no cost).")
    parser.add_argument("--steps", type=int, default=None, help="Override num_steps for a quick pilot run.")
    parser.add_argument("--out", type=str, default="results/results.json")
    parser.add_argument("--only-metric", type=str, default=None, help="Comma-separated: cost,bullwhip")
    parser.add_argument("--only-framework", type=str, default=None,
                         help="Comma-separated: standalone,info_sharing,standalone_tool,info_sharing_tool,negotiation_tool")
    parser.add_argument("--only-model-tier", type=str, default=None, help="Comma-separated: small,large")
    parser.add_argument("--skip-baselines", action="store_true", help="Exclude non-LLM baseline configs from this shard.")
    parser.add_argument("--merge", nargs="+", default=None, help="Merge these shard result JSON files instead of running anything.")
    args = parser.parse_args()

    if args.merge:
        paths = []
        for pattern in args.merge:
            paths.extend(sorted(glob.glob(pattern)))
        if not paths:
            print(f"No files matched: {args.merge}", file=sys.stderr)
            sys.exit(1)
        print(f"Merging {len(paths)} shard file(s): {paths}")
        results = merge_result_files(paths, args.out)
        print()
        print(format_all_tables(results))
        print()
        print(f"Merged {len(results)} result(s) into {args.out}")
        return

    if not args.mock and not os.environ.get(cfg.OPENROUTER_API_KEY_ENV_VAR):
        print(
            f"ERROR: {cfg.OPENROUTER_API_KEY_ENV_VAR} is not set, and --mock was not passed.\n"
            "Either set the API key (real run, incurs API cost) or pass --mock (free pipeline validation only, "
            "results are not meaningful).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.steps is not None:
        cfg.FIXED_PARAMS["num_steps"] = args.steps
        print(f"NOTE: overriding num_steps to {args.steps} for a pilot run (paper uses 200).")

    grid = build_full_grid()
    grid = filter_grid(
        grid,
        only_metric=args.only_metric.split(",") if args.only_metric else None,
        only_framework=args.only_framework.split(",") if args.only_framework else None,
        only_model_tier=args.only_model_tier.split(",") if args.only_model_tier else None,
        skip_baselines=args.skip_baselines,
    )
    if not grid:
        print("ERROR: filters produced an empty config grid.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Running {len(grid)} configuration(s)...")
    results = run_full_grid(use_mock=args.mock, out_path=args.out, grid=grid)

    print()
    print(format_all_tables(results))
    print()
    print(f"Raw results saved to {args.out}")


if __name__ == "__main__":
    main()

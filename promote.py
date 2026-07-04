"""Promote a pipeline run to production: score.py uses it by default.

Usage:
    python promote.py reports/run_xxx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.helpers import get_logger

logger = get_logger("PROMOTE")

PRODUCTION_POINTER = os.path.join("reports", "PRODUCTION")

REQUIRED_ARTIFACTS = [
    os.path.join("model", "model_meta.json"),
    os.path.join("model", "woe_encoder.pkl"),
    os.path.join("strategy", "policy.json"),
]


def promote(run_dir: str) -> str:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    missing = [a for a in REQUIRED_ARTIFACTS if not os.path.exists(os.path.join(run_dir, a))]
    if missing:
        raise FileNotFoundError(
            f"run is not promotable, missing artifacts: {missing}; run pipeline.py to completion first"
        )
    previous = None
    if os.path.exists(PRODUCTION_POINTER):
        with open(PRODUCTION_POINTER, "r", encoding="utf-8") as f:
            previous = f.read().strip() or None
    os.makedirs(os.path.dirname(PRODUCTION_POINTER), exist_ok=True)
    with open(PRODUCTION_POINTER, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(run_dir) + "\n")
    if previous:
        logger.info(f"production run switched: {previous} -> {os.path.abspath(run_dir)}")
    else:
        logger.info(f"production run set: {os.path.abspath(run_dir)}")
    return PRODUCTION_POINTER


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a pipeline run to production")
    parser.add_argument("run_dir", help="Pipeline output directory, e.g. reports/run_xxx")
    args = parser.parse_args()
    promote(args.run_dir)

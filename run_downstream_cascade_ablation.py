#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascade ablation: train CD4 and/or IM from different EL checkpoints.

Edit VARIANTS below to match your EL `name` in yaml and the checkpoint suffix
that actually exists under weights/EL/ (fine-tune-b vs pretrain).

Uses dongyizhe extension entry points (EL cascade CLI). Examples:
  python scripts/run_downstream_cascade_ablation.py --cd4-only --start-id 0 --num-models 1
  python scripts/run_downstream_cascade_ablation.py --im-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Repo root = parent of scripts/
ROOT = Path(__file__).resolve().parents[1]

# --- Edit this list for your EL ablation names (must match weights on disk) ---
VARIANTS: list[dict] = [
    {
        "weights_tag": "from-EL-full",
        "el_stem": "ImmuScope-EL",
        "el_suffix": "fine-tune-b",
    },
    {
        "weights_tag": "from-EL-A2-pretrain",
        "el_stem": "ImmuScope-EL-no-si-no-ftb",
        "el_suffix": "pretrain",
    },
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Batch CD4/IM training over EL variants.")
    p.add_argument("--data-cnf", default="configs/data.yaml")
    p.add_argument("--cd4-cnf", default="configs/ImmuScope.yaml")
    p.add_argument("--im-cnf", default="configs/ImmuScope-IM.yaml")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-models", type=int, default=10)
    p.add_argument("--cd4-only", action="store_true")
    p.add_argument("--im-only", action="store_true")
    args = p.parse_args()

    do_cd4 = not args.im_only
    do_im = not args.cd4_only
    if args.im_only and args.cd4_only:
        p.error("use at most one of --cd4-only / --im-only")

    py = sys.executable
    common = [
        "--data-cnf",
        args.data_cnf,
        "-s",
        str(args.start_id),
        "-n",
        str(args.num_models),
    ]

    for v in VARIANTS:
        tag = v["weights_tag"]
        el_stem = v["el_stem"]
        el_suffix = v["el_suffix"]
        extra = [
            "--el-stem",
            el_stem,
            "--el-suffix",
            el_suffix,
            "--weights-tag",
            tag,
        ]
        if do_cd4:
            run(
                [
                    py,
                    str(ROOT / "main_cd4_epitope_train_dongyizhe_20260405.py"),
                    *common,
                    "-m",
                    args.cd4_cnf,
                    *extra,
                ]
            )
        if do_im:
            run(
                [
                    py,
                    str(ROOT / "main_immunogenicity_train_dongyizhe_20260405.py"),
                    *common,
                    "-m",
                    args.im_cnf,
                    *extra,
                ]
            )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Merge two ImmuScope-style IM training .h5 files (SinInstanceBag format).

Typical use (§2): concatenate official imm_train.h5 with extra IEDB-derived train,
dedupe by (mhc, peptide, context), prefer official rows; then point data.yaml
train_imm to the merged file and keep test_imm as official imm_test.h5.

  python scripts/merge_imm_train_h5.py \\
    --official data/im_datasets/imm_train.h5 \\
    --extra data/iedb_clean/ag85a_tcell_train.h5 \\
    -o data/im_datasets/imm_train_merged_official_iedb.h5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ImmuScope.utils.data_utils import restore_peptide_sequences, save_mhc_peptide_h5py


def _decode_str_array(raw) -> list[str]:
    out = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _load_rows(path: Path) -> list[tuple[str, str, str, float]]:
    """Return list of (mhc, peptide, context, label)."""
    rows: list[tuple[str, str, str, float]] = []
    with h5py.File(path, "r") as f:
        mhc_names = _decode_str_array(f["mhc_names"][()])
        peptide_contexts = _decode_str_array(f["peptide_contexts"][()])
        emb = f["peptide_embedding"][()]
        labels = np.asarray(f["labels"][()])
        if emb.ndim == 3:
            emb = np.squeeze(emb, axis=1)
        peptides = restore_peptide_sequences(emb)
        n = len(labels)
        for i in range(n):
            lab = labels[i]
            if hasattr(lab, "item"):
                lab = float(lab.item())
            else:
                lab = float(lab)
            rows.append((mhc_names[i], peptides[i], peptide_contexts[i], lab))
    return rows


def _dedupe_prefer_first(
    ordered_blocks: list[list[tuple[str, str, str, float]]],
) -> list[tuple[str, str, str, float]]:
    seen: set[tuple[str, str, str]] = set()
    merged: list[tuple[str, str, str, float]] = []
    for block in ordered_blocks:
        for mhc, pep, ctx, lab in block:
            key = (mhc, pep, ctx)
            if key in seen:
                continue
            seen.add(key)
            merged.append((mhc, pep, ctx, lab))
    return merged


def main():
    p = argparse.ArgumentParser(description="Merge IM train .h5 (official + extra, optional dedupe)")
    p.add_argument("--official", type=Path, required=True, help="Primary train .h5 (kept on duplicate)")
    p.add_argument("--extra", type=Path, required=True, help="Additional train .h5 to append")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output merged .h5 path")
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Concatenate all rows even if (mhc, peptide, context) repeats",
    )
    p.add_argument(
        "--label-dtype",
        choices=("float32", "int32"),
        default="float32",
        help="Label dtype in output h5",
    )
    args = p.parse_args()

    for path in (args.official, args.extra):
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")

    official_rows = _load_rows(args.official)
    extra_rows = _load_rows(args.extra)

    if args.no_dedupe:
        merged_tuples = official_rows + extra_rows
    else:
        merged_tuples = _dedupe_prefer_first([official_rows, extra_rows])

    ld = np.float32 if args.label_dtype == "float32" else np.int32
    if args.label_dtype == "int32":
        data = [(a, b, c, int(round(float(d)))) for a, b, c, d in merged_tuples]
    else:
        data = [(a, b, c, float(d)) for a, b, c, d in merged_tuples]

    os.makedirs(args.output.parent, exist_ok=True)
    save_mhc_peptide_h5py(data, str(args.output), label_dtype=ld)

    print(
        f"official rows: {len(official_rows)} | extra rows: {len(extra_rows)} | "
        f"merged: {len(merged_tuples)} | wrote {args.output}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build real context12 (left6 + right6) from NetMHCIIpan-style H5 + FASTA.

Why this script:
- In some H5 files, `peptide_contexts` stores protein identifiers (not full protein sequences).
- MixMHC2pred context mode expects a 12-aa context string for each peptide.

This script maps each H5 sample to its source protein sequence from FASTA, then extracts:
    context12 = left_flank(6) + right_flank(6)
around peptide occurrence in the source protein sequence.

Outputs:
1) CSV with columns:
   mhc_names, peptide, context12, label, protein_id, found_in_protein, match_index
2) Optional MixMHC2pred two-column input:
   peptide<space>context12

Example:
python scripts/build_context12_from_h5.py \
  --h5 data/train_test_h5py/NetMHCIIpan_eval.h5 \
  --fasta data/raw/NetMHCIIpan_eval.fa \
  --out-csv data/tmp_datasets/mixmhc2pred_input_with_context12.csv \
  --out-two-col data/tmp_datasets/mixmhc2pred_input_two_col.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from ImmuScope.utils.data_utils import restore_peptide_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate context12 from H5 and FASTA protein sequences")
    p.add_argument("--h5", type=Path, required=True, help="Input H5 file")
    p.add_argument("--fasta", type=Path, required=True, help="FASTA containing source protein sequences")
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    p.add_argument(
        "--out-two-col",
        type=Path,
        default=None,
        help="Optional output text for MixMHC2pred: 'peptide context12'",
    )
    p.add_argument("--flank", type=int, default=6, help="Flank size on each side (default 6 => context12)")
    p.add_argument(
        "--first-match-only",
        action="store_true",
        help="If peptide appears multiple times in protein, keep first match only (default behavior).",
    )
    return p.parse_args()


def read_fasta(fasta_path: Path) -> Dict[str, str]:
    """
    Parse FASTA and build a sequence dictionary with multiple key styles.
    Key styles:
    - full first token in header (e.g., tr|Q9FPR0|Q9FPR0_POAPR)
    - full header line (without '>')
    """
    seqs: Dict[str, str] = {}
    header = None
    chunks: List[str] = []

    with fasta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(chunks).upper()
                    token = header.split()[0]
                    seqs[token] = seq
                    seqs[header] = seq
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        seq = "".join(chunks).upper()
        token = header.split()[0]
        seqs[token] = seq
        seqs[header] = seq

    return seqs


def normalize_protein_id(raw_id: str) -> str:
    """
    H5 protein_id often looks like: '0_tr|Q9FPR0|Q9FPR0_POAPR'
    Normalize to: 'tr|Q9FPR0|Q9FPR0_POAPR'
    """
    s = raw_id.strip()
    s = re.sub(r"^\d+_", "", s)
    return s


def extract_context12(peptide: str, protein_seq: str, flank: int) -> Tuple[str, int]:
    """
    Return (context, match_index). context length is 2*flank.
    If not found, return ("X"*2*flank, -1).
    """
    pep = peptide.upper()
    seq = protein_seq.upper()
    idx = seq.find(pep)
    if idx < 0:
        return "X" * (2 * flank), -1

    left = seq[max(0, idx - flank) : idx]
    right = seq[idx + len(pep) : idx + len(pep) + flank]

    if len(left) < flank:
        left = ("X" * (flank - len(left))) + left
    if len(right) < flank:
        right = right + ("X" * (flank - len(right)))

    return left + right, idx


def main() -> None:
    args = parse_args()

    if not args.h5.exists():
        raise FileNotFoundError(f"H5 not found: {args.h5}")
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")

    fasta_map = read_fasta(args.fasta)
    if not fasta_map:
        raise RuntimeError(f"No sequences parsed from FASTA: {args.fasta}")

    with h5py.File(args.h5, "r") as f:
        emb = f["peptide_embedding"][()]
        mhc_raw = f["mhc_names"][()]
        labels = np.asarray(f["labels"][()], dtype=float)
        protein_raw = f["peptide_contexts"][()]

    peptides = np.array(restore_peptide_sequences(emb, peptide_pad=3))
    mhc = np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in mhc_raw])
    protein_ids = np.array(
        [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in protein_raw]
    )

    rows = []
    found = 0
    not_found = 0
    missing_protein = 0

    for a, pep, lab, pid in zip(mhc, peptides, labels, protein_ids):
        pid_norm = normalize_protein_id(pid)

        # Try direct key, then first token style
        seq = fasta_map.get(pid_norm)
        if seq is None:
            seq = fasta_map.get(pid_norm.split()[0])

        if seq is None:
            context12 = "X" * (2 * args.flank)
            idx = -2  # protein id not found in FASTA
            missing_protein += 1
            not_found += 1
        else:
            context12, idx = extract_context12(pep, seq, args.flank)
            if idx >= 0:
                found += 1
            else:
                not_found += 1

        rows.append(
            {
                "mhc_names": a,
                "peptide": pep,
                "context12": context12,
                "label": float(lab),
                "protein_id": pid_norm,
                "found_in_protein": int(idx >= 0),
                "match_index": int(idx),
            }
        )

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    if args.out_two_col is not None:
        args.out_two_col.parent.mkdir(parents=True, exist_ok=True)
        with args.out_two_col.open("w", encoding="utf-8") as fw:
            for _, r in out_df.iterrows():
                fw.write(f"{r['peptide']} {r['context12']}\n")

    print(f"[OK] wrote CSV: {args.out_csv}")
    if args.out_two_col is not None:
        print(f"[OK] wrote two-column input: {args.out_two_col}")
    print(f"[INFO] total={len(out_df)} found={found} not_found={not_found} missing_protein={missing_protein}")


if __name__ == "__main__":
    main()

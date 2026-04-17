# -*- coding: utf-8 -*-
"""
Clean IEDB "T cell assay" table export (.xlsx / .csv) and map to ImmuScope .h5.

Expected columns (IEDB full export, single or double header):
  - Epitope - Name: peptide sequence
  - Epitope - Object Type: keep "Linear peptide"
  - MHC Restriction - Name: allele (e.g. HLA-DRB1*01:01)
  - Assay - Qualitative Measurement: Positive / Negative / ...
  - Epitope - Source Molecule (optional context; falls back to Assay Antigen - Source Molecule)

Run from repo root:
  # IEDB "Full" Excel exports often use two header rows → add: --header-rows 2
  python scripts/iedb_tcell_export_to_h5.py \\
    -i data/raw/iedb/tcell_table_export_1775714140.xlsx --header-rows 2 \\
    --mhc-seq data/raw/pseudosequence.2023.dat \\
    -o data/iedb_clean/tcell_clean.tsv \\
    --h5 data/iedb_h5/tcell_eval.h5

  Train/test split (writes prefix_train.tsv / prefix_test.tsv and optional .h5):
  python scripts/iedb_tcell_export_to_h5.py -i ... --filter-known-mhc \\
    --split-train-ratio 0.8 --split-prefix data/iedb_clean/tcell --split-h5

Requires: pandas; for .xlsx also: pip install openpyxl
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ImmuScope.utils.aminoacids import INVALID_ACIDS
from ImmuScope.utils.data_utils import get_mhc_name_seq, save_mhc_peptide_h5py

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("Please install pandas: pip install pandas") from e


def _norm_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def _pick_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Match column by normalized exact name or substring."""
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        nc = _norm_col(cand)
        if nc in norm_map:
            return norm_map[nc]
    for cand in candidates:
        key = _norm_col(cand)
        for c in df.columns:
            if key in _norm_col(c):
                return c
    return None


def _read_table(path: Path, header_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if header_rows == 1:
        kw = {"header": 0}
    else:
        kw = {"header": list(range(header_rows))}
    if suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path, engine="openpyxl", **kw)
        except ImportError as e:
            raise SystemExit(
                "Reading .xlsx needs openpyxl: pip install openpyxl"
            ) from e
    if suffix == ".csv":
        return pd.read_csv(path, **kw)
    raise SystemExit(f"Unsupported file type: {suffix}")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    cols = []
    for tup in df.columns.values:
        parts = [str(p).strip() for p in tup if str(p).strip().lower() not in ("nan", "none", "")]
        cols.append(" - ".join(parts) if len(parts) > 1 else (parts[0] if parts else "col"))
    out = df.copy()
    out.columns = cols
    return out


def _valid_peptide(seq: str, max_len: int) -> bool:
    if not seq or len(seq) > max_len:
        return False
    for ch in seq.upper():
        if ch in INVALID_ACIDS:
            return False
    return bool(seq)


def _map_qualitative(x) -> int | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    t = str(x).strip().lower()
    if not t:
        return None
    if t in ("positive", "pos") or "positive" == t:
        return 1
    if t in ("negative", "neg") or "negative" == t:
        return 0
    if "positive" in t and "negative" not in t:
        return 1
    if "negative" in t and "positive" not in t:
        return 0
    return None


def _split_train_test(
    rows: list[tuple],
    train_ratio: float,
    seed: int,
    by_context: bool,
) -> tuple[list[tuple], list[tuple]]:
    """Split rows into train and test. Optionally group by context (no shared context)."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return [], []
    if n == 1:
        return rows, []

    if by_context:
        ctx_to_idx: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            ctx_to_idx.setdefault(r[2], []).append(i)
        contexts = list(ctx_to_idx.keys())
        if len(contexts) == 1:
            raise SystemExit(
                "split-by-context needs at least 2 distinct context values; got 1."
            )
        rng.shuffle(contexts)
        n_train_ctx = max(1, int(round(len(contexts) * train_ratio)))
        n_train_ctx = min(n_train_ctx, len(contexts) - 1)
        train_ctx = set(contexts[:n_train_ctx])
        train_rows = [rows[i] for i in range(n) if rows[i][2] in train_ctx]
        test_rows = [rows[i] for i in range(n) if rows[i][2] not in train_ctx]
        return train_rows, test_rows

    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, min(n - 1, int(round(n * train_ratio))))
    train_set = set(idx[:n_train].tolist())
    train_rows = [rows[i] for i in range(n) if i in train_set]
    test_rows = [rows[i] for i in range(n) if i not in train_set]
    return train_rows, test_rows


def normalize_iedb_mhc_to_pseudo_key(name: str) -> str:
    """
    Map common IEDB allele strings to keys used in pseudosequence.2023.dat.

    Examples:
      HLA-DRB1*01:01 -> DRB1_0101
      HLA-DQA1*05:01/DQB1*02:01 -> HLA-DQA10501-DQB10201
    If no rule matches, returns the stripped original string.
    """
    s = name.strip()
    if not s:
        return s

    # Single-chain DRB (most common in IEDB class II)
    m = re.match(r"^HLA-(DRB[1-5])\*(\d{2}):(\d{2})$", s, re.I)
    if m:
        g1, g2, g3 = m.group(1).upper(), m.group(2), m.group(3)
        return f"{g1}_{g2}{g3}"

    # DRB without HLA- prefix
    m = re.match(r"^(DRB[1-5])\*(\d{2}):(\d{2})$", s, re.I)
    if m:
        g1, g2, g3 = m.group(1).upper(), m.group(2), m.group(3)
        return f"{g1}_{g2}{g3}"

    # DQ heterodimer: HLA-DQA1*aa:bb / DQB1*cc:dd
    m = re.match(
        r"^HLA-(DQA1)\*(\d{2}):(\d{2})\s*/\s*(DQB1)\*(\d{2}):(\d{2})$",
        s,
        re.I,
    )
    if m:
        _, a1, a2, _, b1, b2 = m.groups()
        return f"HLA-DQA1{a1}{a2}-DQB1{b1}{b2}"

    # DP heterodimer
    m = re.match(
        r"^HLA-(DPA1)\*(\d{2}):(\d{2})\s*/\s*(DPB1)\*(\d{2}):(\d{2})$",
        s,
        re.I,
    )
    if m:
        _, a1, a2, _, b1, b2 = m.groups()
        return f"HLA-DPA1{a1}{a2}-DPB1{b1}{b2}"

    return s


def _is_unspecific_mhc(name: str) -> bool:
    n = name.strip().lower()
    if not n:
        return True
    vague = (
        "class ii",
        "class 2",
        "mhc class ii",
        "hla class ii",
        "ii restriction",
    )
    return any(v in n for v in vague) and "*" not in name


def main():
    p = argparse.ArgumentParser(description="IEDB T cell export -> TSV + optional ImmuScope .h5")
    p.add_argument("-i", "--input", type=Path, required=True, help="IEDB export .xlsx or .csv")
    p.add_argument("--header-rows", type=int, default=1, choices=(1, 2), help="1=single row header; 2=IEDB double header")
    p.add_argument(
        "--mhc-seq",
        type=Path,
        default=_ROOT / "data/raw/pseudosequence.2023.dat",
        help="Pseudo sequence file; rows with unknown alleles are dropped if --filter-known-mhc",
    )
    p.add_argument("--filter-known-mhc", action="store_true", help="Keep only MHC names present in --mhc-seq")
    p.add_argument(
        "--no-normalize-mhc",
        action="store_true",
        help="Do not map IEDB names (e.g. HLA-DRB1*01:01) to pseudosequence keys (e.g. DRB1_0101). "
        "Without normalization, --filter-known-mhc often keeps 0 rows.",
    )
    p.add_argument("--drop-unspecific-mhc", action="store_true", default=True, help="Drop rows like 'HLA class II' without allele")
    p.add_argument("--keep-unspecific-mhc", action="store_true", help="Opposite of --drop-unspecific-mhc")
    p.add_argument("--max-peptide-len", type=int, default=21, help="Max peptide length (ImmuScope EL/CD4 default is 21)")
    p.add_argument(
        "--truncate-long-peptides",
        action="store_true",
        help="If peptide longer than --max-peptide-len, keep first N residues instead of dropping",
    )
    p.add_argument("-o", "--output-tsv", type=Path, default=None, help="Write cleaned four-column TSV (full set)")
    p.add_argument("--h5", type=Path, default=None, help="Write ImmuScope .h5 via save_mhc_peptide_h5py (full set)")
    p.add_argument("--label-dtype", choices=("float32", "int32"), default="float32")
    p.add_argument(
        "--split-train-ratio",
        type=float,
        default=None,
        help="If set (e.g. 0.8), split into train/test; requires --split-prefix",
    )
    p.add_argument(
        "--split-prefix",
        type=Path,
        default=None,
        help="Path prefix without extension: writes {prefix}_train.tsv and {prefix}_test.tsv",
    )
    p.add_argument("--split-seed", type=int, default=2024, help="RNG seed for --split-train-ratio")
    p.add_argument(
        "--split-by-context",
        action="store_true",
        help="Assign whole context groups to train or test (less leakage than random rows)",
    )
    p.add_argument(
        "--split-h5",
        action="store_true",
        help="With split, also write {prefix}_train.h5 and {prefix}_test.h5",
    )
    args = p.parse_args()

    if args.keep_unspecific_mhc:
        args.drop_unspecific_mhc = False

    if args.split_train_ratio is not None:
        if not (0.0 < args.split_train_ratio < 1.0):
            raise SystemExit("--split-train-ratio must be between 0 and 1 (exclusive).")
        if args.split_prefix is None:
            raise SystemExit("--split-train-ratio requires --split-prefix (path without extension).")

    df = _read_table(args.input, args.header_rows)
    df = _flatten_columns(df)

    col_pep = _pick_column(df, "Epitope - Name", "epitope - name")
    col_lin = _pick_column(df, "Epitope - Object Type", "epitope - object type")
    col_mhc = _pick_column(df, "MHC Restriction - Name", "mhc restriction - name")
    col_qual = _pick_column(df, "Assay - Qualitative Measurement", "assay - qualitative measurement")
    col_ctx1 = _pick_column(df, "Epitope - Source Molecule", "epitope - source molecule")
    col_ctx2 = _pick_column(df, "Assay Antigen - Source Molecule", "assay antigen - source molecule")

    missing = [n for n, c in [
        ("Epitope - Name", col_pep),
        ("MHC Restriction - Name", col_mhc),
        ("Assay - Qualitative Measurement", col_qual),
    ] if c is None]
    if missing:
        raise SystemExit(
            "Missing columns: %s. Available columns (first 30): %s"
            % (missing, list(df.columns[:30]))
        )

    mhc_name_seq = {}
    if args.filter_known_mhc:
        if not args.mhc_seq.exists():
            raise SystemExit(f"--mhc-seq not found: {args.mhc_seq}")
        mhc_name_seq = get_mhc_name_seq(str(args.mhc_seq))

    do_normalize_mhc = not args.no_normalize_mhc

    rows_out = []
    stats = {
        "n_in": len(df),
        "n_not_linear": 0,
        "n_bad_peptide": 0,
        "n_unspecific_mhc": 0,
        "n_unknown_allele": 0,
        "n_unknown_label": 0,
        "n_out": 0,
    }

    for _, r in df.iterrows():
        if col_lin is not None:
            ot = r[col_lin]
            if pd.notna(ot) and str(ot).strip().lower() != "linear peptide":
                stats["n_not_linear"] += 1
                continue

        raw_pep = r[col_pep]
        if pd.isna(raw_pep):
            stats["n_bad_peptide"] += 1
            continue
        pep = re.sub(r"\s+", "", str(raw_pep)).upper()
        if args.truncate_long_peptides and len(pep) > args.max_peptide_len:
            pep = pep[: args.max_peptide_len]
        if not _valid_peptide(pep, args.max_peptide_len):
            stats["n_bad_peptide"] += 1
            continue

        mhc = str(r[col_mhc]).strip() if pd.notna(r[col_mhc]) else ""
        if args.drop_unspecific_mhc and _is_unspecific_mhc(mhc):
            stats["n_unspecific_mhc"] += 1
            continue

        if do_normalize_mhc:
            mhc = normalize_iedb_mhc_to_pseudo_key(mhc)

        if args.filter_known_mhc and mhc_name_seq and mhc not in mhc_name_seq:
            stats["n_unknown_allele"] += 1
            continue

        lab = _map_qualitative(r[col_qual])
        if lab is None:
            stats["n_unknown_label"] += 1
            continue

        ctx = ""
        for cc in (col_ctx1, col_ctx2):
            if cc is not None and pd.notna(r[cc]):
                ctx = str(r[cc]).strip()
                break
        if not ctx:
            ctx = "NA"

        rows_out.append((mhc, pep, ctx, float(lab)))

    stats["n_out"] = len(rows_out)

    print("IEDB T cell clean summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    ld = np.float32 if args.label_dtype == "float32" else np.int32

    if args.split_train_ratio is not None:
        train_rows, test_rows = _split_train_test(
            rows_out,
            args.split_train_ratio,
            args.split_seed,
            args.split_by_context,
        )
        prefix = args.split_prefix
        os.makedirs(prefix.parent, exist_ok=True)
        train_tsv = prefix.parent / f"{prefix.name}_train.tsv"
        test_tsv = prefix.parent / f"{prefix.name}_test.tsv"
        pd.DataFrame(train_rows, columns=["mhc_name", "peptide", "context", "label"]).to_csv(
            train_tsv, sep="\t", index=False
        )
        pd.DataFrame(test_rows, columns=["mhc_name", "peptide", "context", "label"]).to_csv(
            test_tsv, sep="\t", index=False
        )
        print(f"Wrote split TSV: {train_tsv} ({len(train_rows)} rows), {test_tsv} ({len(test_rows)} rows)")
        if args.split_h5:
            train_h5 = prefix.parent / f"{prefix.name}_train.h5"
            test_h5 = prefix.parent / f"{prefix.name}_test.h5"
            save_mhc_peptide_h5py(train_rows, str(train_h5), label_dtype=ld)
            save_mhc_peptide_h5py(test_rows, str(test_h5), label_dtype=ld)
            print(f"Wrote split H5: {train_h5}, {test_h5}")

    if args.output_tsv:
        out_df = pd.DataFrame(rows_out, columns=["mhc_name", "peptide", "context", "label"])
        os.makedirs(args.output_tsv.parent, exist_ok=True)
        out_df.to_csv(args.output_tsv, sep="\t", index=False)
        print(f"Wrote TSV: {args.output_tsv}")

    if args.h5:
        os.makedirs(args.h5.parent, exist_ok=True)
        save_mhc_peptide_h5py(rows_out, str(args.h5), label_dtype=ld)
        print(f"Wrote H5: {args.h5}")

    if (
        not args.output_tsv
        and not args.h5
        and args.split_train_ratio is None
    ):
        print("No -o/--h5 given; only printed summary. Add -o or --h5 to write files.")


if __name__ == "__main__":
    main()

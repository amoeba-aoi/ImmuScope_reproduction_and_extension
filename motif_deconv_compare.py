#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motif deconvolution and comparative evaluation for ImmuScope / NetMHCIIpan / MixMHC2pred.

Input format (CSV for each method):
  Required columns:
    - mhc_names   : allele (e.g., DRB1_0401)
    - peptide     : peptide sequence
    - pred        : model score (higher = stronger binder)

Outputs:
  1) per-method per-allele PWM CSVs
  2) logo figures (if logomaker installed)
  3) pairwise motif similarity table (JSD/KL)
  4) summary table aggregated across alleles
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_LIST = list(AA)
AA_SET = set(AA_LIST)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Motif deconvolution and cross-method motif comparison")
    p.add_argument("--immuscope-csv", type=Path, required=True)
    p.add_argument("--netmhc-csv", type=Path, required=True)
    p.add_argument("--mixmhc-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/motif_compare"))
    p.add_argument("--core-len", type=int, default=9)
    p.add_argument("--top-k", type=int, default=500, help="Top peptides per allele per method")
    p.add_argument("--min-samples", type=int, default=100, help="Min peptides to keep an allele")
    p.add_argument("--pseudocount", type=float, default=1e-3)
    p.add_argument("--logo-dpi", type=int, default=180)
    return p.parse_args()


def _check_columns(df: pd.DataFrame, path: Path) -> None:
    needed = {"mhc_names", "peptide", "pred"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing required columns: {sorted(miss)}")


def read_method_csv(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, path)
    df = df[["mhc_names", "peptide", "pred"]].copy()
    df["method"] = method
    df["mhc_names"] = df["mhc_names"].astype(str)
    df["peptide"] = df["peptide"].astype(str).str.upper()
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["pred"])
    df = df[df["peptide"].map(lambda x: len(x) > 0 and set(x).issubset(AA_SET))]
    return df


def pick_core_by_window(peptide: str, core_len: int) -> str:
    """
    Lightweight core alignment:
    choose the best window by a heuristic anchor score (hydrophobic + aromatic preference).
    """
    if len(peptide) <= core_len:
        return peptide.ljust(core_len, "A")

    anchor_bonus = set("FWYVILM")
    best_score = -1e9
    best = peptide[:core_len]
    for i in range(0, len(peptide) - core_len + 1):
        w = peptide[i : i + core_len]
        score = 0.0
        # Soft anchor prior: emphasize positions around p1/p4/p6/p9
        for pos in (0, 3, 5, core_len - 1):
            if w[pos] in anchor_bonus:
                score += 1.0
        # mild complexity preference
        score += len(set(w)) * 0.05
        if score > best_score:
            best_score = score
            best = w
    return best


def build_pwm(cores: Iterable[str], core_len: int, pseudocount: float) -> pd.DataFrame:
    arr = np.full((core_len, len(AA_LIST)), pseudocount, dtype=np.float64)
    aa_index = {aa: i for i, aa in enumerate(AA_LIST)}
    n = 0
    for c in cores:
        if len(c) != core_len:
            continue
        if not set(c).issubset(AA_SET):
            continue
        n += 1
        for i, ch in enumerate(c):
            arr[i, aa_index[ch]] += 1.0
    arr /= arr.sum(axis=1, keepdims=True)
    pwm = pd.DataFrame(arr, columns=AA_LIST)
    pwm.index = [f"P{i+1}" for i in range(core_len)]
    pwm.attrs["n_cores"] = n
    return pwm


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def motif_distance(pwm_a: pd.DataFrame, pwm_b: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns:
      jsd_mean, kl_ab_mean, kl_ba_mean averaged over positions.
    """
    jsd_vals, kl_ab_vals, kl_ba_vals = [], [], []
    for i in range(len(pwm_a)):
        pa = pwm_a.iloc[i].to_numpy(dtype=np.float64)
        pb = pwm_b.iloc[i].to_numpy(dtype=np.float64)
        jsd_vals.append(jsd(pa, pb))
        kl_ab_vals.append(kl(pa, pb))
        kl_ba_vals.append(kl(pb, pa))
    return float(np.mean(jsd_vals)), float(np.mean(kl_ab_vals)), float(np.mean(kl_ba_vals))


def save_logo(pwm: pd.DataFrame, title: str, out_png: Path, dpi: int = 180) -> None:
    try:
        import matplotlib.pyplot as plt
        import logomaker
    except Exception:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    # Convert probability to information-like height
    prob = pwm.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore"):
        entropy = -np.sum(prob * np.log2(np.clip(prob, 1e-12, 1.0)), axis=1)
    info = np.maximum(0.0, math.log2(20.0) - entropy)
    heights = prob * info[:, None]
    # logomaker requires integer row indices (1-based positions); not "P1","P2",...
    hdf = pd.DataFrame(
        heights, columns=AA_LIST, index=range(1, len(heights) + 1)
    )
    logomaker.Logo(hdf, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Core position")
    ax.set_ylabel("Bits")
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pwm_dir = out_dir / "pwms"
    logo_dir = out_dir / "logos"
    pwm_dir.mkdir(parents=True, exist_ok=True)
    logo_dir.mkdir(parents=True, exist_ok=True)

    df_imm = read_method_csv(args.immuscope_csv, "ImmuScope")
    df_net = read_method_csv(args.netmhc_csv, "NetMHCIIpan")
    df_mix = read_method_csv(args.mixmhc_csv, "MixMHC2pred")
    all_df = pd.concat([df_imm, df_net, df_mix], ignore_index=True)

    # Keep alleles present in all methods with enough examples
    counts = (
        all_df.groupby(["method", "mhc_names"])["peptide"]
        .size()
        .reset_index(name="n")
    )
    ok_alleles = []
    for allele in sorted(all_df["mhc_names"].unique()):
        sub = counts[counts["mhc_names"] == allele]
        if len(sub) != 3:
            continue
        if (sub["n"] >= args.min_samples).all():
            ok_alleles.append(allele)

    if not ok_alleles:
        raise RuntimeError("No allele passed filters; lower --min-samples or check inputs.")

    method_pwms: Dict[Tuple[str, str], pd.DataFrame] = {}
    compare_rows: List[Dict[str, object]] = []

    for allele in ok_alleles:
        for method in ("ImmuScope", "NetMHCIIpan", "MixMHC2pred"):
            sub = all_df[(all_df["method"] == method) & (all_df["mhc_names"] == allele)].copy()
            sub = sub.sort_values("pred", ascending=False).head(args.top_k)
            cores = [pick_core_by_window(p, args.core_len) for p in sub["peptide"].tolist()]
            pwm = build_pwm(cores, args.core_len, args.pseudocount)
            method_pwms[(method, allele)] = pwm

            pwm_file = pwm_dir / f"{method}__{allele}__pwm.csv"
            pwm.to_csv(pwm_file, index=True)
            save_logo(
                pwm,
                title=f"{method} | {allele} | n={pwm.attrs.get('n_cores', 0)}",
                out_png=logo_dir / f"{method}__{allele}.png",
                dpi=args.logo_dpi,
            )

        # pairwise comparisons
        pairs = [
            ("ImmuScope", "NetMHCIIpan"),
            ("ImmuScope", "MixMHC2pred"),
            ("NetMHCIIpan", "MixMHC2pred"),
        ]
        for a, b in pairs:
            pwm_a = method_pwms[(a, allele)]
            pwm_b = method_pwms[(b, allele)]
            jsd_mean, kl_ab, kl_ba = motif_distance(pwm_a, pwm_b)
            compare_rows.append(
                {
                    "allele": allele,
                    "method_a": a,
                    "method_b": b,
                    "jsd_mean": jsd_mean,
                    "kl_a_to_b_mean": kl_ab,
                    "kl_b_to_a_mean": kl_ba,
                    "n_core_a": pwm_a.attrs.get("n_cores", 0),
                    "n_core_b": pwm_b.attrs.get("n_cores", 0),
                }
            )

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_dir / "motif_pairwise_similarity.csv", index=False)

    summary = (
        compare_df.groupby(["method_a", "method_b"], as_index=False)[
            ["jsd_mean", "kl_a_to_b_mean", "kl_b_to_a_mean"]
        ]
        .mean()
        .sort_values(["method_a", "method_b"])
    )
    summary.to_csv(out_dir / "motif_similarity_summary.csv", index=False)

    print(f"[OK] Alleles processed: {len(ok_alleles)}")
    print(f"[OK] Wrote: {out_dir / 'motif_pairwise_similarity.csv'}")
    print(f"[OK] Wrote: {out_dir / 'motif_similarity_summary.csv'}")
    print(f"[OK] PWM dir:  {pwm_dir}")
    print(f"[OK] Logo dir: {logo_dir}")
    print("[NOTE] If logo files are missing, install matplotlib/logomaker.")


if __name__ == "__main__":
    main()

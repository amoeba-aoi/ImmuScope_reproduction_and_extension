#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.1–3.6 推断统计辅助脚本：bootstrap / Friedman / Wilcoxon 等；在有成对分数处补充 **AUPR** 与阈值 **PPV**（默认 0.5）。

运行（在项目根）:
  python scripts/inferential_stats_sections.py \\
    --reproduction-dir ... --ablation-dir ... --project-root ...

输出目录: results/inferential_stats/

3.2 与 `scripts/compare_cd4_auc_three.py` 的 **intersection** 内连接一致（acc + dup_id），
在 **三方法交集** 上做样本级配对 bootstrap；另对 **同一 per-protein AUC** 做 **Wilcoxon 符号秩**（与 protein median bootstrap 同一 merge）。
3.3 在存在 **allele, jsd_mean** 列时，对 **同一 allele 上两两 JSD 之差** 做可选 Wilcoxon + Holm。
3.6（可选，`--partition-variant-root` 或默认可探测 `ImmuScope_change_division`）：**数据划分变体** 全链路 vs **reproduction**；CD4/IM 同 3.1；EL 为 per-allele 探索性检验（见 JSON caveat）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    average_precision_score = None  # type: ignore
    roc_auc_score = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "inferential_stats"

# PPV = precision among predicted-positive at fixed score threshold (same for all sections using PPV).
PPV_DEFAULT_THRESHOLD = 0.5


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    b = np.where(y > 0.5, 1, 0)
    if len(np.unique(b)) < 2:
        return float("nan")
    return float(roc_auc_score(b, s))


def _aupr(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (area under precision–recall curve), sklearn positive class = 1."""
    y = np.asarray(y, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    b = np.where(y > 0.5, 1, 0)
    if len(np.unique(b)) < 2:
        return float("nan")
    return float(average_precision_score(b, s))


def _ppv(y: np.ndarray, s: np.ndarray, threshold: float = PPV_DEFAULT_THRESHOLD) -> float:
    """PPV at fixed threshold: TP / (TP + FP) for predictions score >= threshold."""
    y = np.asarray(y, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    yb = np.where(y > 0.5, 1, 0)
    if len(np.unique(yb)) < 2:
        return float("nan")
    pred_pos = (s >= threshold).astype(int)
    tp = int(np.sum((pred_pos == 1) & (yb == 1)))
    fp = int(np.sum((pred_pos == 1) & (yb == 0)))
    if tp + fp == 0:
        return float("nan")
    return float(tp / (tp + fp))


def _stratified_bootstrap_idx(y: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """保证子样本中正负例均存在（各至少 1），避免 AUC=nan。"""
    y = np.asarray(y, dtype=float)
    pos = np.where(y > 0.5)[0]
    neg = np.where(y <= 0.5)[0]
    if len(pos) == 0 or len(neg) == 0:
        return rng.integers(0, len(y), size=m)
    n_pos = max(1, int(round(m * len(pos) / len(y))))
    n_pos = min(n_pos, m - 1)
    n_neg = m - n_pos
    n_pos = max(1, n_pos)
    n_neg = max(1, n_neg)
    if n_pos + n_neg > m:
        n_neg = m - n_pos
    ip = rng.choice(pos, size=n_pos, replace=True)
    in_ = rng.choice(neg, size=n_neg, replace=True)
    return np.concatenate([ip, in_])


def _bootstrap_paired_scalar_metric_diff(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
    boot_size_cap: int | None = 100_000,
) -> dict:
    """
    配对样本级 bootstrap：**Δ = metric(s1) − metric(s2)** 在同一重抽子集上计算。
    与 `bootstrap_auc_diff_paired` 同一重抽策略（含超大 n 分层子样本）。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    n = len(y)
    m = n if boot_size_cap is None else min(n, boot_size_cap)
    obs = float(metric_fn(y, s1) - metric_fn(y, s2))
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if boot_size_cap is not None and n > boot_size_cap:
            idx = _stratified_bootstrap_idx(y, m, rng)
        else:
            idx = rng.integers(0, n, size=m)
        diffs[b] = float(metric_fn(y[idx], s1[idx]) - metric_fn(y[idx], s2[idx]))
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_two = min(1.0, p_two)
    return {
        "n_samples": int(n),
        "bootstrap_draw_size_m": int(m),
        "observed_delta": float(obs),
        "bootstrap_B": int(n_boot),
        "ci95_delta": [float(ci_lo), float(ci_hi)],
        "p_value_two_sided_bootstrap": float(p_two),
    }


def bootstrap_auc_diff_paired(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    stat: str = "auc_diff",
    boot_size_cap: int | None = 100_000,
) -> dict:
    """
    配对单位：**样本级**（同一测试实例上两个模型的分数）。
    过程：有放回重抽 **样本索引** B 次；每次计算 AUC(s1)-AUC(s2)。
    若 n > boot_size_cap，每次重抽 **固定长度 m=min(n,cap)** 的分层子样本。
    """
    r = _bootstrap_paired_scalar_metric_diff(
        y, s1, s2, _auc, n_boot=n_boot, seed=seed, boot_size_cap=boot_size_cap
    )
    return {
        "n_samples": r["n_samples"],
        "bootstrap_draw_size_m": r["bootstrap_draw_size_m"],
        "observed_delta_auc": r["observed_delta"],
        "bootstrap_B": r["bootstrap_B"],
        "ci95_delta_auc": r["ci95_delta"],
        "p_value_two_sided_bootstrap": r["p_value_two_sided_bootstrap"],
        "statistic": stat,
    }


def bootstrap_aupr_diff_paired(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    stat: str = "aupr_diff",
    boot_size_cap: int | None = 100_000,
) -> dict:
    """同 `bootstrap_auc_diff_paired`，度量换为 **AUPR**（average precision）。"""
    r = _bootstrap_paired_scalar_metric_diff(
        y, s1, s2, _aupr, n_boot=n_boot, seed=seed, boot_size_cap=boot_size_cap
    )
    return {
        "n_samples": r["n_samples"],
        "bootstrap_draw_size_m": r["bootstrap_draw_size_m"],
        "observed_delta_aupr": r["observed_delta"],
        "bootstrap_B": r["bootstrap_B"],
        "ci95_delta_aupr": r["ci95_delta"],
        "p_value_two_sided_bootstrap": r["p_value_two_sided_bootstrap"],
        "statistic": stat,
        "metric": "average_precision_AUPR",
    }


def bootstrap_ppv_diff_paired(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    *,
    threshold: float = PPV_DEFAULT_THRESHOLD,
    n_boot: int = 2000,
    seed: int = 42,
    stat: str = "ppv_diff",
    boot_size_cap: int | None = 100_000,
) -> dict:
    """同 `bootstrap_auc_diff_paired`，度量换为 **PPV**（score >= threshold 的 precision）。"""

    def _ppv_bind(ya: np.ndarray, sa: np.ndarray) -> float:
        return _ppv(ya, sa, threshold)

    r = _bootstrap_paired_scalar_metric_diff(
        y, s1, s2, _ppv_bind, n_boot=n_boot, seed=seed, boot_size_cap=boot_size_cap
    )
    return {
        "n_samples": r["n_samples"],
        "bootstrap_draw_size_m": r["bootstrap_draw_size_m"],
        "observed_delta_ppv": r["observed_delta"],
        "bootstrap_B": r["bootstrap_B"],
        "ci95_delta_ppv": r["ci95_delta"],
        "p_value_two_sided_bootstrap": r["p_value_two_sided_bootstrap"],
        "statistic": stat,
        "metric": "PPV_at_threshold",
        "ppv_threshold": float(threshold),
    }


def bootstrap_median_protein_auc_diff(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    protein_col: str = "Protein",
    auc_col: str = "AUC",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    配对单位：**protein 级** AUC（每个 protein 一个 AUC，两模型 join 后逐 protein 成对）。
    """
    rng = np.random.default_rng(seed)
    m = df1.merge(df2, on=protein_col, suffixes=("_a", "_b"))
    if len(m) == 0:
        return {
            "n_proteins": 0,
            "observed_median_delta_auc": float("nan"),
            "bootstrap_B": int(n_boot),
            "ci95_median_delta": [float("nan"), float("nan")],
            "p_value_two_sided_bootstrap": float("nan"),
            "error": "empty_merge",
        }
    a = m[f"{auc_col}_a"].to_numpy(dtype=float)
    b = m[f"{auc_col}_b"].to_numpy(dtype=float)
    p = len(m)
    obs = float(np.median(a - b))
    diffs = np.empty(n_boot, dtype=float)
    idx_all = np.arange(p)
    for i in range(n_boot):
        j = rng.choice(idx_all, size=p, replace=True)
        diffs[i] = float(np.median(a[j] - b[j]))
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_two = min(1.0, p_two)
    return {
        "n_proteins": int(p),
        "observed_median_delta_auc": obs,
        "bootstrap_B": int(n_boot),
        "ci95_median_delta": [float(ci_lo), float(ci_hi)],
        "p_value_two_sided_bootstrap": float(p_two),
    }


def wilcoxon_protein_auc_diff(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    protein_col: str = "Protein",
    auc_col: str = "AUC",
) -> dict:
    """
    与 `bootstrap_median_protein_auc_diff` 相同 merge：**逐 protein** 指标差 d = col(df1) − col(df2)；
    对 d 做 **双侧 Wilcoxon 符号秩**（与 3.4 事后同一 scipy 调用）。
    """
    m = df1.merge(df2, on=protein_col, suffixes=("_a", "_b"))
    if len(m) == 0:
        return {
            "n_proteins": 0,
            "median_delta_auc": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "p_value_two_sided": float("nan"),
            "error": "empty_merge",
        }
    a = m[f"{auc_col}_a"].to_numpy(dtype=float)
    b = m[f"{auc_col}_b"].to_numpy(dtype=float)
    d = a - b
    base = {
        "n_proteins": int(len(m)),
        "pairing": "paired_per_protein_on_intersection_long_table",
        "unit": f"protein_level_{auc_col.lower()}_diff_wilcoxon_signed_rank",
        "delta_definition": (
            f"{auc_col}(first_df) - {auc_col}(second_df); "
            "first/second order matches protein_median_bootstrap block"
        ),
    }
    if np.allclose(d, 0):
        return {
            **base,
            "median_delta_auc": 0.0,
            "wilcoxon_statistic": 0.0,
            "p_value_two_sided": 1.0,
            "note": "all_protein_auc_differences_zero",
        }
    w, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {
        **base,
        "median_delta_auc": float(np.median(d)),
        "wilcoxon_statistic": float(w),
        "p_value_two_sided": float(p),
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm 校正（与 R p.adjust(..., 'holm') 等价）。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sp = p[order]
    mult = np.arange(m, 0, -1)  # m, m-1, ..., 1
    adj_sorted = np.maximum.accumulate(mult * sp)
    inv = np.argsort(order)
    adj = np.minimum(adj_sorted[inv], 1.0)
    return [float(x) for x in adj]


def accession_from_protein(s: str) -> str:
    """与 scripts/compare_cd4_auc_three.py 一致：protein 串按 | 分割后的第二段（UniProt id）。"""
    segs = str(s).split("|")
    if len(segs) >= 3:
        return segs[1]
    return str(s)[:40]


def add_acc_dup_id_im_nm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out["acc"] = out["protein"].astype(str).map(accession_from_protein)
    out["dup_id"] = out.groupby(["mhc_names", "peptide", "acc"], sort=False).cumcount()
    return out


def add_acc_dup_id_mix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    if "protein_id" not in out.columns:
        raise ValueError("MixMHC2pred CSV 需要列 protein_id")
    out["acc"] = out["protein_id"].astype(str).map(accession_from_protein)
    out["dup_id"] = out.groupby(["mhc_names", "peptide", "acc"], sort=False).cumcount()
    return out


def load_cd4_intersection_three(proj: Path) -> pd.DataFrame:
    """
    与 `compare_cd4_auc_three.py --mode intersection` 相同的三表内连接：
      - Imm ∩ Net：键 (mhc_names, peptide, protein, label, acc, dup_id)
      - 再 ∩ Mix：键 (mhc_names, peptide, acc, dup_id) + pred_mix
    得到 **同一测试实例** 上 pred_im / pred_nm / pred_mix，用于样本级配对推断。
    """
    im_p = proj / "results/ImmuScope-CD4/results_pred_per_sample.csv"
    nm_p = proj / "results/NetMHCIIpan-CD4/results_pred_per_sample.csv"
    mx_p = proj / "results/MixMHC2pred-CD4/results_pred_per_sample.csv"
    im = pd.read_csv(im_p)
    nm = pd.read_csv(nm_p)
    mx = pd.read_csv(mx_p)
    im = add_acc_dup_id_im_nm(im)
    nm = add_acc_dup_id_im_nm(nm)
    mx = add_acc_dup_id_mix(mx)
    merged = im.merge(
        nm,
        on=["mhc_names", "peptide", "protein", "label", "acc", "dup_id"],
        how="inner",
        suffixes=("_im", "_nm"),
    )
    mx_pred = mx[["mhc_names", "peptide", "acc", "dup_id", "pred"]].rename(
        columns={"pred": "pred_mix"}
    )
    merged = merged.merge(mx_pred, on=["mhc_names", "peptide", "acc", "dup_id"], how="inner")
    need = {"pred_im", "pred_nm", "pred_mix", "label"}
    if not need.issubset(merged.columns):
        raise ValueError(f"交集列异常，需 {need}，当前: {merged.columns.tolist()}")
    return merged


def per_protein_auc_from_merged(merged: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return per_protein_metric_from_merged(merged, pred_col, _auc, "AUC")


def per_protein_metric_from_merged(
    merged: pd.DataFrame,
    pred_col: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    out_col: str,
) -> pd.DataFrame:
    rows = []
    for prot, sub in merged.groupby("protein"):
        y = sub["label"].to_numpy(dtype=float)
        s = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(s) & np.isfinite(y)
        if len(np.unique(np.where(y[m] > 0.5, 1, 0))) < 2:
            continue
        v = metric_fn(y[m], s[m])
        if np.isfinite(v):
            rows.append({"Protein": str(prot), out_col: float(v)})
    return pd.DataFrame(rows)


def cd4_paired_protein_metric_tables(
    df_rep: pd.DataFrame, df_orig: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None:
    """
    两表行序对齐时，逐 protein 计算 AUC / AUPR / PPV（PPV 使用 `PPV_DEFAULT_THRESHOLD`）。
    返回各度量的 (df_rep, df_orig)，列名为 AUC / AUPR / PPV。
    """
    if len(df_rep) != len(df_orig):
        return None
    r_rep_auc: list[dict] = []
    r_orig_auc: list[dict] = []
    r_rep_aupr: list[dict] = []
    r_orig_aupr: list[dict] = []
    r_rep_ppv: list[dict] = []
    r_orig_ppv: list[dict] = []
    for prot, idx in df_rep.groupby("protein", sort=False).groups.items():
        sa = df_rep.loc[idx].reset_index(drop=True)
        sb = df_orig.loc[idx].reset_index(drop=True)
        if len(sa) != len(sb):
            continue
        y_a = sa["label"].to_numpy(dtype=float)
        y_b = sb["label"].to_numpy(dtype=float)
        if not (y_a == y_b).all():
            continue
        p_a = pd.to_numeric(sa["pred"], errors="coerce").to_numpy(dtype=float)
        p_b = pd.to_numeric(sb["pred"], errors="coerce").to_numpy(dtype=float)
        if len(np.unique(np.where(y_a > 0.5, 1, 0))) < 2:
            continue
        auc_rep = _auc(y_a, p_a)
        auc_orig = _auc(y_a, p_b)
        if np.isfinite(auc_rep) and np.isfinite(auc_orig):
            r_rep_auc.append({"Protein": str(prot), "AUC": float(auc_rep)})
            r_orig_auc.append({"Protein": str(prot), "AUC": float(auc_orig)})
        aupr_rep = _aupr(y_a, p_a)
        aupr_orig = _aupr(y_a, p_b)
        if np.isfinite(aupr_rep) and np.isfinite(aupr_orig):
            r_rep_aupr.append({"Protein": str(prot), "AUPR": float(aupr_rep)})
            r_orig_aupr.append({"Protein": str(prot), "AUPR": float(aupr_orig)})
        ppv_rep = _ppv(y_a, p_a)
        ppv_orig = _ppv(y_a, p_b)
        if np.isfinite(ppv_rep) and np.isfinite(ppv_orig):
            r_rep_ppv.append({"Protein": str(prot), "PPV": float(ppv_rep)})
            r_orig_ppv.append({"Protein": str(prot), "PPV": float(ppv_orig)})
    if not r_rep_auc:
        return None
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "AUC": (pd.DataFrame(r_rep_auc), pd.DataFrame(r_orig_auc)),
    }
    if r_rep_aupr:
        out["AUPR"] = (pd.DataFrame(r_rep_aupr), pd.DataFrame(r_orig_aupr))
    if r_rep_ppv:
        out["PPV"] = (pd.DataFrame(r_rep_ppv), pd.DataFrame(r_orig_ppv))
    return out


def cd4_paired_protein_auc_tables(
    df_rep: pd.DataFrame, df_orig: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """向后兼容：仅返回 (Protein, AUC) 对表。"""
    d = cd4_paired_protein_metric_tables(df_rep, df_orig)
    if d is None:
        return None
    return d["AUC"]


def bootstrap_median_allele_auc01_diff(
    d: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    d = 逐 allele 的 ΔAUC0.1（与 Wilcoxon 同一向量）。对 **allele 行** 有放回重抽，重复算 median(d)。
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {
            "n_alleles": 0,
            "observed_median_delta": float("nan"),
            "bootstrap_B": int(n_boot),
            "ci95_median_delta": [float("nan"), float("nan")],
            "p_value_two_sided_bootstrap": float("nan"),
            "error": "empty",
        }
    obs = float(np.median(d))
    diffs = np.empty(n_boot, dtype=float)
    idx_all = np.arange(n)
    for i in range(n_boot):
        j = rng.choice(idx_all, size=n, replace=True)
        diffs[i] = float(np.median(d[j]))
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_two = min(1.0, p_two)
    return {
        "n_alleles": int(n),
        "observed_median_delta": obs,
        "bootstrap_B": int(n_boot),
        "ci95_median_delta": [float(ci_lo), float(ci_hi)],
        "p_value_two_sided_bootstrap": float(p_two),
        "pairing": "paired_by_allele_name_inner_merge",
        "unit": "allele_level_median_delta_auc01",
    }


def wilcoxon_paired_allele_auc01_diff(merged: pd.DataFrame, d: np.ndarray) -> dict:
    """同一 merged 上的逐 allele ΔAUC0.1；双侧 Wilcoxon 符号秩。"""
    base = {
        "n_alleles": int(len(merged)),
        "pairing": "paired_by_allele_name_inner_merge",
        "unit": "allele_level_delta_auc01_wilcoxon_signed_rank",
        "delta_definition": "AUC0.1(partition_variant) - AUC0.1(reproduction)",
    }
    if len(d) == 0:
        return {**base, "p_value_two_sided": float("nan"), "error": "empty"}
    if np.allclose(d, 0):
        return {
            **base,
            "median_delta_auc01": 0.0,
            "wilcoxon_statistic": 0.0,
            "p_value_two_sided": 1.0,
            "note": "all_allele_auc_differences_zero",
        }
    w, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {
        **base,
        "median_delta_auc01": float(np.median(d)),
        "mean_delta_auc01": float(np.mean(d)),
        "wilcoxon_statistic": float(w),
        "p_value_two_sided": float(p),
    }


def section_partition_vs_reproduction(partition_root: Path, reproduction_root: Path) -> dict:
    """
    3.6：EL **数据划分变体**（如 train_ratio 调整）完整重训链路 vs **默认划分的 reproduction**。

    - CD4 / IM：与 3.1 相同配对逻辑，**ΔAUC = AUC(变体) − AUC(reproduction)**；样本级 bootstrap（CD4 大样本用分层子样本）。
    - EL：仅有 per-allele 汇总 CSV 时，在 **allele 名** 上内连接后对 **ΔAUC0.1** 做 **Wilcoxon** 与 **allele 维度 bootstrap median**。
      若两跑的 **total（该 allele 评估肽数）不一致**，则各 allele 的 AUC 基于**不同实例集合**，此时 EL 检验为 **探索性**（见 JSON 中 caveat）。
    """
    part = partition_root.resolve()
    rep = reproduction_root.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(rep.resolve()))
        except ValueError:
            return str(p.resolve())

    res: dict = {
        "section": "3.6",
        "comparison": (
            "Data partitioning variant (full retrain: EL + downstream CD4/IM) vs "
            "ImmuScope_reproduction default split pipeline"
        ),
        "delta_auc_definition": "AUC(partition_variant) - AUC(reproduction)",
        "why_methods_en": (
            "For **CD4** and **IM**, predictions are available on the **same fixed test instances** "
            "(row-aligned CD4 `results_pred_protein_avg.csv`; IM inner-joined on mhc, peptide, label), "
            "so **paired nonparametric bootstrap** on ΔAUC matches Section 3.1: it targets uncertainty "
            "conditional on the two fixed checkpoints without retraining noise. "
            "For **EL**, only **per-allele** summaries (`AUC0.1`, `total`) are compared; when `total` "
            "differs between runs, each allele’s AUC is computed on **different peptide sets**, so "
            "**Wilcoxon / allele-bootstrap** on ΔAUC0.1 are **exploratory** summaries of cross-allele "
            "direction of change, **not** a paired-instance test on identical ligands."
        ),
        "why_methods_zh": (
            "CD4、IM 均在**同一固定测试实例**上有成对分数（CD4 与 3.1 一样行对齐；IM 在 mhc+peptide+label 上合并），"
            "故采用与 3.1 一致的 **配对、有放回 bootstrap** 推断 **ΔAUC = AUC(划分变体) − AUC(reproduction)**，"
            "度量在**两枚固定 checkpoint** 下整体排序差异，不含重训随机性。"
            "EL 仅有 **按 allele 汇总** 的 AUC0.1 与 total；若两次运行的 **total 不同**，则各 allele 的 AUC 基于**不同肽集合**，"
            "此时对 ΔAUC0.1 的 **Wilcoxon / 按 allele bootstrap** 仅作 **探索性** 跨 allele 方向汇总，**不能**视为同一批肽上的配对实例检验。"
        ),
        "notes": [],
    }

    cd4_p = part / "results/ImmuScope-CD4/results_pred_protein_avg.csv"
    cd4_r = rep / "results/ImmuScope-CD4/results_pred_protein_avg.csv"
    res["cd4_paths"] = {
        "partition_variant": str(cd4_p),
        "reproduction": _rel(cd4_r),
    }
    if cd4_p.is_file() and cd4_r.is_file():
        a = pd.read_csv(cd4_p)
        b = pd.read_csv(cd4_r)
        if len(a) != len(b):
            res["cd4"] = {"error": "row count mismatch"}
        elif not (a["label"].to_numpy() == b["label"].to_numpy()).all():
            res["cd4"] = {"error": "label mismatch by row"}
        else:
            res["cd4"] = bootstrap_auc_diff_paired(
                a["label"].to_numpy(),
                a["pred"].to_numpy(),
                b["pred"].to_numpy(),
                n_boot=800,
                boot_size_cap=100_000,
            )
            res["cd4"]["pairing"] = "paired_by_row_index_same_test_order"
            res["cd4"]["unit"] = "sample_level_instances"
            res["cd4"]["comparison"] = (
                "ImmuScope-CD4 (partition variant EL downstream) vs ImmuScope-CD4 (reproduction)"
            )
            sb_ca = bootstrap_aupr_diff_paired(
                a["label"].to_numpy(),
                a["pred"].to_numpy(),
                b["pred"].to_numpy(),
                n_boot=800,
                boot_size_cap=100_000,
                stat="delta_aupr_variant_minus_repro",
            )
            sb_ca["pairing"] = res["cd4"]["pairing"]
            sb_ca["unit"] = "sample_level_aupr_delta"
            sb_ca["comparison"] = res["cd4"]["comparison"]
            res["cd4"]["sample_bootstrap_aupr"] = sb_ca
            sb_cp = bootstrap_ppv_diff_paired(
                a["label"].to_numpy(),
                a["pred"].to_numpy(),
                b["pred"].to_numpy(),
                n_boot=800,
                boot_size_cap=100_000,
                stat="delta_ppv_variant_minus_repro",
            )
            sb_cp["pairing"] = res["cd4"]["pairing"]
            sb_cp["unit"] = "sample_level_ppv_delta"
            sb_cp["comparison"] = res["cd4"]["comparison"]
            res["cd4"]["sample_bootstrap_ppv"] = sb_cp
            mt = cd4_paired_protein_metric_tables(a, b)
            if mt is not None:
                d_part, d_rep = mt["AUC"]
                med = bootstrap_median_protein_auc_diff(
                    d_part, d_rep, n_boot=5000, seed=42
                )
                med["pairing"] = "paired_per_protein_same_row_aligned_subtables"
                med["unit"] = "protein_level_median_auc_delta_variant_minus_repro"
                med["note"] = (
                    "Each protein: AUC computed on the same rows in both CSVs; "
                    "bootstrap resamples proteins with replacement. "
                    "Delta = AUC(variant) - AUC(reproduction)."
                )
                res["cd4"]["protein_median_bootstrap"] = med
                if "AUPR" in mt:
                    dp, dr = mt["AUPR"]
                    ma = bootstrap_median_protein_auc_diff(
                        dp, dr, auc_col="AUPR", n_boot=5000, seed=42
                    )
                    ma["pairing"] = med["pairing"]
                    ma["unit"] = "protein_level_median_aupr_delta_variant_minus_repro"
                    res["cd4"]["protein_median_bootstrap_aupr"] = ma
                if "PPV" in mt:
                    pp, pr = mt["PPV"]
                    mp = bootstrap_median_protein_auc_diff(
                        pp, pr, auc_col="PPV", n_boot=5000, seed=42
                    )
                    mp["pairing"] = med["pairing"]
                    mp["unit"] = "protein_level_median_ppv_delta_variant_minus_repro"
                    res["cd4"]["protein_median_bootstrap_ppv"] = mp

    im_p = part / "results/ImmuScope-IM/results_ImmuScope-IM_avg.csv"
    im_r = rep / "results/ImmuScope-IM/results_ImmuScope-IM_avg.csv"
    res["im_paths"] = {"partition_variant": str(im_p), "reproduction": _rel(im_r)}
    if im_p.is_file() and im_r.is_file():
        a = pd.read_csv(im_p)
        b = pd.read_csv(im_r)
        keys = ["mhc", "peptide", "label"]
        m = a.merge(b, on=keys, suffixes=("_part", "_repro"))
        res["im"] = bootstrap_auc_diff_paired(
            m["label"].to_numpy(),
            m["pred_part"].to_numpy(),
            m["pred_repro"].to_numpy(),
            n_boot=800,
            boot_size_cap=None,
        )
        res["im"]["pairing"] = "paired_on_mhc_peptide_label"
        res["im"]["unit"] = "sample_level_instances"
        res["im"]["n_merged"] = int(len(m))
        res["im"]["comparison"] = (
            "ImmuScope-IM (partition variant) vs ImmuScope-IM (reproduction)"
        )
        im_ca = bootstrap_aupr_diff_paired(
            m["label"].to_numpy(),
            m["pred_part"].to_numpy(),
            m["pred_repro"].to_numpy(),
            n_boot=800,
            boot_size_cap=None,
            stat="delta_aupr_variant_minus_repro",
        )
        im_ca["pairing"] = res["im"]["pairing"]
        im_ca["unit"] = "sample_level_aupr_delta"
        im_ca["comparison"] = res["im"]["comparison"]
        im_ca["n_merged"] = int(len(m))
        res["im"]["sample_bootstrap_aupr"] = im_ca
        im_cp = bootstrap_ppv_diff_paired(
            m["label"].to_numpy(),
            m["pred_part"].to_numpy(),
            m["pred_repro"].to_numpy(),
            n_boot=800,
            boot_size_cap=None,
            stat="delta_ppv_variant_minus_repro",
        )
        im_cp["pairing"] = res["im"]["pairing"]
        im_cp["unit"] = "sample_level_ppv_delta"
        im_cp["comparison"] = res["im"]["comparison"]
        im_cp["n_merged"] = int(len(m))
        res["im"]["sample_bootstrap_ppv"] = im_cp

    el_p = part / "results/ImmuScope-EL/results_ImmuScope-EL_avg.csv"
    el_r = rep / "results/ImmuScope-EL/results_ImmuScope-EL_avg.csv"
    res["el_paths"] = {"partition_variant": str(el_p), "reproduction": _rel(el_r)}
    if el_p.is_file() and el_r.is_file():
        dp = pd.read_csv(el_p)
        dr = pd.read_csv(el_r)
        mg = dp.merge(dr, on="allele", suffixes=("_part", "_repro"))
        auc_p = mg["AUC0.1_part"].to_numpy(dtype=float)
        auc_r = mg["AUC0.1_repro"].to_numpy(dtype=float)
        tot_p = mg["total_part"].to_numpy(dtype=float)
        tot_r = mg["total_repro"].to_numpy(dtype=float)
        d = auc_p - auc_r
        n_match_total = int(np.sum(tot_p == tot_r))
        fin_w = (
            np.isfinite(tot_p)
            & np.isfinite(tot_r)
            & np.isfinite(auc_p)
            & np.isfinite(auc_r)
            & (tot_p > 0)
            & (tot_r > 0)
        )
        if np.any(fin_w):
            tp, tr = tot_p[fin_w], tot_r[fin_w]
            overall_p = float((auc_p[fin_w] * tp).sum() / tp.sum())
            overall_r = float((auc_r[fin_w] * tr).sum() / tr.sum())
        else:
            overall_p = float("nan")
            overall_r = float("nan")
        el_block: dict = {
            "descriptive": {
                "n_alleles_merged": int(len(mg)),
                "n_alleles_same_total_count": n_match_total,
                "per_allele_totals_all_differ": bool(n_match_total == 0 and len(mg) > 0),
                "n_alleles_used_for_weighted_overall": int(np.sum(fin_w)),
                "n_alleles_excluded_nonfinite_or_nonpositive_total": int(np.sum(~fin_w)),
                "overall_auc01_partition_weighted": overall_p,
                "overall_auc01_reproduction_weighted": overall_r,
                "delta_overall_auc01_heuristic": float(overall_p - overall_r),
                "note_en": (
                    "Weighted overalls use each run’s own `total` weights; they are not a single pooled "
                    "ROC on identical peptides when totals differ."
                ),
                "note_zh": "两行 overall 各自按本侧的 total 加权；当各 allele 的 total 不一致时，二者之差仅为启发式对比，非同一批肽上的单一 pooled AUC 差。",
            },
            "wilcoxon_paired_allele_auc01": wilcoxon_paired_allele_auc01_diff(mg, d),
            "bootstrap_median_delta_allele_auc01": bootstrap_median_allele_auc01_diff(
                d, n_boot=5000, seed=42
            ),
        }
        el_block["interpretation_caveat_en"] = (
            "Per-allele AUC0.1 may be computed on **different peptide sets** when `total` differs; "
            "Wilcoxon and allele-bootstrap p-values are **exploratory** tests on allele-level deltas, "
            "not paired-instance inference on the same ligands."
        )
        el_block["interpretation_caveat_zh"] = (
            "当各 allele 的 `total` 与 reproduction 不一致时，allele 级 AUC0.1 可能基于**不同肽集合**；"
            "Wilcoxon 与按 allele 的 bootstrap p 为对 **allele 级 Δ** 的探索性检验，**不是**同一批肽实例上的配对推断。"
        )
        res["el"] = el_block

    # Holm across CD4 sample, CD4 protein median, IM sample, EL Wilcoxon
    p_named: list[tuple[str, float]] = []
    cd4 = res.get("cd4")
    if isinstance(cd4, dict) and "p_value_two_sided_bootstrap" in cd4:
        p_named.append(("cd4_sample_bootstrap", float(cd4["p_value_two_sided_bootstrap"])))
        pm = cd4.get("protein_median_bootstrap")
        if isinstance(pm, dict) and "p_value_two_sided_bootstrap" in pm:
            p_named.append(
                ("cd4_protein_median_bootstrap", float(pm["p_value_two_sided_bootstrap"]))
            )
        sa = cd4.get("sample_bootstrap_aupr")
        if isinstance(sa, dict) and "p_value_two_sided_bootstrap" in sa:
            p_named.append(("cd4_sample_bootstrap_aupr", float(sa["p_value_two_sided_bootstrap"])))
        sp = cd4.get("sample_bootstrap_ppv")
        if isinstance(sp, dict) and "p_value_two_sided_bootstrap" in sp:
            p_named.append(("cd4_sample_bootstrap_ppv", float(sp["p_value_two_sided_bootstrap"])))
        pma = cd4.get("protein_median_bootstrap_aupr")
        if isinstance(pma, dict) and "p_value_two_sided_bootstrap" in pma:
            p_named.append(
                ("cd4_protein_median_bootstrap_aupr", float(pma["p_value_two_sided_bootstrap"]))
            )
        pmp = cd4.get("protein_median_bootstrap_ppv")
        if isinstance(pmp, dict) and "p_value_two_sided_bootstrap" in pmp:
            p_named.append(
                ("cd4_protein_median_bootstrap_ppv", float(pmp["p_value_two_sided_bootstrap"]))
            )
    imd = res.get("im")
    if isinstance(imd, dict) and "p_value_two_sided_bootstrap" in imd:
        p_named.append(("im_sample_bootstrap", float(imd["p_value_two_sided_bootstrap"])))
        ima = imd.get("sample_bootstrap_aupr")
        if isinstance(ima, dict) and "p_value_two_sided_bootstrap" in ima:
            p_named.append(("im_sample_bootstrap_aupr", float(ima["p_value_two_sided_bootstrap"])))
        imp = imd.get("sample_bootstrap_ppv")
        if isinstance(imp, dict) and "p_value_two_sided_bootstrap" in imp:
            p_named.append(("im_sample_bootstrap_ppv", float(imp["p_value_two_sided_bootstrap"])))
    el = res.get("el")
    if isinstance(el, dict):
        wx = el.get("wilcoxon_paired_allele_auc01")
        if isinstance(wx, dict) and "p_value_two_sided" in wx and "error" not in wx:
            p_named.append(("el_allele_wilcoxon_exploratory", float(wx["p_value_two_sided"])))

    if p_named:
        names, pvals = zip(*p_named, strict=True)
        adj = holm_adjust(list(pvals))
        res["holm_across_endpoints"] = {
            "tests": [
                {"name": n, "p_raw": float(pv), "p_holm": float(a)}
                for n, pv, a in zip(names, pvals, adj, strict=True)
            ],
            "note_en": (
                "Holm step-down across the listed endpoints (now includes **AUPR/PPV** sample- and protein-level "
                "CD4 tests and IM sample-level tests where present). **el_allele_wilcoxon_exploratory** should "
                "still be read with the EL **total-mismatch** caveat."
            ),
            "note_zh": (
                "对下列终点做 Holm 校正（已纳入 **AUPR/PPV** 的 CD4 样本/蛋白级与 IM 样本级检验，若存在）。"
                "**el_allele_wilcoxon_exploratory** 仍须结合 EL **total 不一致** 的探索性局限解读。"
            ),
        }

    res["notes"].append(
        "Section 3.6 inference is conditional on fixed checkpoints; it does not include multi-seed training variance."
    )
    return res


def section_3_1(rep: Path, out: Path) -> dict:
    """
    3.1：同一固定测试集上，**reproduction（EL 下游训练产出）** vs **original-weight 检查点**。

    - CD4：`results_pred_protein_avg.csv` 按 **行序配对**（须与 originalweight 导出一致）；
      补充 **protein 级** median ΔAUC 的 bootstrap（与主文按蛋白汇总口径一致）。
    - IM：`results_ImmuScope-IM_avg.csv` 在 (**mhc, peptide, label**) 上内连接配对。
    统计量均为 **ΔAUC = AUC(reproduction) − AUC(original_weight)**（单次 checkpoint，不含重训随机性）。
    """
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(rep.resolve()))
        except ValueError:
            return str(p.resolve())

    res: dict = {
        "section": "3.1",
        "notes": [],
        "appropriate_methods_zh": (
            "在**同一测试标签、同一实例配对**前提下，用 **ROC-AUC 差（reproduction − original）** 度量系统差异；"
            "并**并列**报告 **AUPR（average precision）** 与 **PPV（默认阈值 0.5）** 的样本级及 protein 级 median 差之 bootstrap。"
            "不确定度均由 **有放回 bootstrap** 得到 CI 与双侧经验 p。"
            "这是**固定测试集 + 固定两枚已训练模型**下的推断；若要包含训练随机性，需多随机种子重复训练再做层级模型。"
        ),
        "delta_auc_definition": "AUC(reproduction_EL_downstream) - AUC(original_weight_checkpoint)",
        "delta_aupr_definition": "AUPR(reproduction) - AUPR(original_weight); AUPR = sklearn average_precision (positive class 1)",
        "delta_ppv_definition": (
            f"PPV(reproduction) - PPV(original_weight) at score >= {PPV_DEFAULT_THRESHOLD}; "
            "PPV = TP/(TP+FP) among predicted-positive"
        ),
    }
    cd4_repro = rep / "results/ImmuScope-CD4/results_pred_protein_avg.csv"
    cd4_orig = rep / "results/ImmuScope-CD4-originalweight/results_pred_protein_avg.csv"
    res["cd4_paths"] = {
        "reproduction_EL_downstream": _rel(cd4_repro),
        "original_weight": _rel(cd4_orig),
    }
    if cd4_repro.is_file() and cd4_orig.is_file():
        a = pd.read_csv(cd4_repro)
        b = pd.read_csv(cd4_orig)
        if len(a) != len(b):
            res["cd4"] = {"error": "row count mismatch"}
        else:
            if not (a["label"].to_numpy() == b["label"].to_numpy()).all():
                res["cd4"] = {"error": "label mismatch by row"}
            else:
                res["cd4"] = bootstrap_auc_diff_paired(
                    a["label"].to_numpy(),
                    a["pred"].to_numpy(),
                    b["pred"].to_numpy(),
                    n_boot=800,
                    boot_size_cap=100_000,
                )
                res["cd4"]["pairing"] = "paired_by_row_index_same_test_order"
                res["cd4"]["unit"] = "sample_level_instances"
                res["cd4"]["comparison"] = (
                    "ImmuScope-CD4 (reproduction EL downstream) vs ImmuScope-CD4-originalweight"
                )
                sb_a = bootstrap_aupr_diff_paired(
                    a["label"].to_numpy(),
                    a["pred"].to_numpy(),
                    b["pred"].to_numpy(),
                    n_boot=800,
                    boot_size_cap=100_000,
                    stat="delta_aupr_repro_minus_orig",
                )
                sb_a["pairing"] = res["cd4"]["pairing"]
                sb_a["unit"] = "sample_level_aupr_delta"
                sb_a["comparison"] = res["cd4"]["comparison"]
                res["cd4"]["sample_bootstrap_aupr"] = sb_a
                sb_p = bootstrap_ppv_diff_paired(
                    a["label"].to_numpy(),
                    a["pred"].to_numpy(),
                    b["pred"].to_numpy(),
                    n_boot=800,
                    boot_size_cap=100_000,
                    stat="delta_ppv_repro_minus_orig",
                )
                sb_p["pairing"] = res["cd4"]["pairing"]
                sb_p["unit"] = "sample_level_ppv_delta"
                sb_p["comparison"] = res["cd4"]["comparison"]
                res["cd4"]["sample_bootstrap_ppv"] = sb_p
                mt = cd4_paired_protein_metric_tables(a, b)
                if mt is not None:
                    d_rep, d_orig = mt["AUC"]
                    med = bootstrap_median_protein_auc_diff(
                        d_rep, d_orig, n_boot=5000, seed=42
                    )
                    med["pairing"] = "paired_per_protein_same_row_aligned_subtables"
                    med["unit"] = "protein_level_median_auc_delta_repro_minus_orig"
                    med["note"] = (
                        "Each protein: AUC computed on the same rows in both CSVs; "
                        "bootstrap resamples proteins with replacement."
                    )
                    res["cd4"]["protein_median_bootstrap"] = med
                    if "AUPR" in mt:
                        dr, do = mt["AUPR"]
                        ma = bootstrap_median_protein_auc_diff(
                            dr, do, auc_col="AUPR", n_boot=5000, seed=42
                        )
                        ma["pairing"] = med["pairing"]
                        ma["unit"] = "protein_level_median_aupr_delta_repro_minus_orig"
                        ma["note"] = (
                            "Per-protein AUPR on the same row-aligned slices; "
                            "bootstrap over proteins."
                        )
                        res["cd4"]["protein_median_bootstrap_aupr"] = ma
                    if "PPV" in mt:
                        pr, po = mt["PPV"]
                        mp = bootstrap_median_protein_auc_diff(
                            pr, po, auc_col="PPV", n_boot=5000, seed=42
                        )
                        mp["pairing"] = med["pairing"]
                        mp["unit"] = "protein_level_median_ppv_delta_repro_minus_orig"
                        mp["note"] = (
                            f"Per-protein PPV at threshold {PPV_DEFAULT_THRESHOLD}; "
                            "bootstrap over proteins."
                        )
                        res["cd4"]["protein_median_bootstrap_ppv"] = mp
    im_repro = rep / "results/ImmuScope-IM/results_ImmuScope-IM_avg.csv"
    im_orig = rep / "results/ImmuScope-IM-originalweight/results_ImmuScope-IM_avg.csv"
    res["im_paths"] = {
        "reproduction_EL_downstream": _rel(im_repro),
        "original_weight": _rel(im_orig),
    }
    if im_repro.is_file() and im_orig.is_file():
        a = pd.read_csv(im_repro)
        b = pd.read_csv(im_orig)
        keys = ["mhc", "peptide", "label"]
        m = a.merge(b, on=keys, suffixes=("_repro", "_orig"))
        res["im"] = bootstrap_auc_diff_paired(
            m["label"].to_numpy(),
            m["pred_repro"].to_numpy(),
            m["pred_orig"].to_numpy(),
            n_boot=800,
            boot_size_cap=None,
        )
        res["im"]["pairing"] = "paired_on_mhc_peptide_label"
        res["im"]["unit"] = "sample_level_instances"
        res["im"]["n_merged"] = int(len(m))
        res["im"]["comparison"] = (
            "ImmuScope-IM (reproduction EL downstream) vs ImmuScope-IM-originalweight"
        )
        im_a = bootstrap_aupr_diff_paired(
            m["label"].to_numpy(),
            m["pred_repro"].to_numpy(),
            m["pred_orig"].to_numpy(),
            n_boot=800,
            boot_size_cap=None,
            stat="delta_aupr_repro_minus_orig",
        )
        im_a["pairing"] = res["im"]["pairing"]
        im_a["unit"] = "sample_level_aupr_delta"
        im_a["comparison"] = res["im"]["comparison"]
        im_a["n_merged"] = int(len(m))
        res["im"]["sample_bootstrap_aupr"] = im_a
        im_p = bootstrap_ppv_diff_paired(
            m["label"].to_numpy(),
            m["pred_repro"].to_numpy(),
            m["pred_orig"].to_numpy(),
            n_boot=800,
            boot_size_cap=None,
            stat="delta_ppv_repro_minus_orig",
        )
        im_p["pairing"] = res["im"]["pairing"]
        im_p["unit"] = "sample_level_ppv_delta"
        im_p["comparison"] = res["im"]["comparison"]
        im_p["n_merged"] = int(len(m))
        res["im"]["sample_bootstrap_ppv"] = im_p
    res["notes"].append(
        "Replication inference only quantifies uncertainty on the fixed test sample; "
        "it does not replace multi-seed training variance."
    )
    return res


def section_3_2(proj: Path, out: Path) -> dict:
    """
    3.2：与 `results/compare_cd4_auc_three.csv`（intersection 模式）一致，
    在 **三表交集样本** 上做 **样本级配对** bootstrap ΔAUC；两两比较共 3 次，p 值 **Holm 校正**。
    另：在交集长表上按 **protein** 重算 per-protein AUC，对逐蛋白 ΔAUC 做 **Wilcoxon 符号秩**（`wilcoxon_on_protein_auc_diff`）
    与 **median Δ 的 bootstrap**（`protein_median_bootstrap`），各 3 次比较均 **Holm**。
    """
    res = {"section": "3.2"}
    merged = load_cd4_intersection_three(proj)
    y = pd.to_numeric(merged["label"], errors="coerce").to_numpy(dtype=float)
    pi = pd.to_numeric(merged["pred_im"], errors="coerce").to_numpy(dtype=float)
    pn = pd.to_numeric(merged["pred_nm"], errors="coerce").to_numpy(dtype=float)
    pm = pd.to_numeric(merged["pred_mix"], errors="coerce").to_numpy(dtype=float)
    m_ok = np.isfinite(y) & np.isfinite(pi) & np.isfinite(pn) & np.isfinite(pm)
    merged_ok = merged.loc[m_ok].reset_index(drop=True)
    y, pi, pn, pm = y[m_ok], pi[m_ok], pn[m_ok], pm[m_ok]

    res["intersection_protocol"] = (
        "Same as scripts/compare_cd4_auc_three.py intersection: "
        "Imm∩Net on (mhc_names, peptide, protein, label, acc, dup_id); "
        "then ∩ Mix on (mhc_names, peptide, acc, dup_id); "
        "acc=2nd pipe field of protein; dup_id=within-group window index."
    )
    res["n_intersection_samples"] = int(len(y))

    pairs = [
        ("ImmuScope_vs_NetMHCIIpan", pi, pn, "delta_auc_ImmuScope_minus_Net"),
        ("ImmuScope_vs_MixMHC2pred", pi, pm, "delta_auc_ImmuScope_minus_Mix"),
        ("NetMHCIIpan_vs_MixMHC2pred", pn, pm, "delta_auc_Net_minus_Mix"),
    ]
    res["pairwise_sample_bootstrap"] = {}
    raw_p_sample = []
    labels_s = []
    for name, s1, s2, stat_label in pairs:
        r = bootstrap_auc_diff_paired(
            y, s1, s2, n_boot=800, boot_size_cap=100_000, stat=stat_label
        )
        r["pairing"] = "paired_same_intersection_instance_three_methods"
        r["unit"] = "sample_level_auc_delta"
        r["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["pairwise_sample_bootstrap"][name] = r
        raw_p_sample.append(r["p_value_two_sided_bootstrap"])
        labels_s.append(name)
    adj_s = holm_adjust(raw_p_sample)
    for name, ap in zip(labels_s, adj_s):
        res["pairwise_sample_bootstrap"][name]["p_holm"] = float(ap)

    df_i = per_protein_auc_from_merged(merged_ok, "pred_im")
    df_n = per_protein_auc_from_merged(merged_ok, "pred_nm")
    df_m = per_protein_auc_from_merged(merged_ok, "pred_mix")
    res["protein_median_bootstrap"] = {}
    raw_p_prot = []
    labels_p = []
    for name, a, b in [
        ("ImmuScope_vs_NetMHCIIpan", df_i, df_n),
        ("ImmuScope_vs_MixMHC2pred", df_i, df_m),
        ("NetMHCIIpan_vs_MixMHC2pred", df_n, df_m),
    ]:
        rp = bootstrap_median_protein_auc_diff(a, b, n_boot=5000)
        rp["pairing"] = "paired_per_protein_on_intersection_long_table"
        rp["unit"] = "protein_level_median_auc_delta"
        rp["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["protein_median_bootstrap"][name] = rp
        raw_p_prot.append(rp["p_value_two_sided_bootstrap"])
        labels_p.append(name)
    # Holm 跳过 nan p（空 merge）
    valid_idx = [i for i, p in enumerate(raw_p_prot) if np.isfinite(p)]
    if valid_idx:
        sub_p = [raw_p_prot[i] for i in valid_idx]
        sub_adj = holm_adjust(sub_p)
        j = 0
        for i in range(len(labels_p)):
            if i in valid_idx:
                res["protein_median_bootstrap"][labels_p[i]]["p_holm"] = float(sub_adj[j])
                j += 1
            else:
                res["protein_median_bootstrap"][labels_p[i]]["p_holm"] = float("nan")

    # 与上同一 per-protein AUC 表：Wilcoxon 符号秩检验「逐蛋白 AUC 差」是否系统偏离 0（对齐原文 Fig.2 蛋白层思路）
    prot_pairs = [
        ("ImmuScope_vs_NetMHCIIpan", df_i, df_n),
        ("ImmuScope_vs_MixMHC2pred", df_i, df_m),
        ("NetMHCIIpan_vs_MixMHC2pred", df_n, df_m),
    ]
    res["wilcoxon_on_protein_auc_diff"] = {}
    raw_w: list[float] = []
    names_w: list[str] = []
    for name, a, b in prot_pairs:
        wr = wilcoxon_protein_auc_diff(a, b)
        wr["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["wilcoxon_on_protein_auc_diff"][name] = wr
        pv = wr.get("p_value_two_sided", float("nan"))
        if "error" not in wr and np.isfinite(pv):
            raw_w.append(float(pv))
            names_w.append(name)
    if raw_w:
        adj_w = holm_adjust(raw_w)
        for nm, ap in zip(names_w, adj_w):
            res["wilcoxon_on_protein_auc_diff"][nm]["p_holm"] = float(ap)
    for name, _, _ in prot_pairs:
        if "p_holm" not in res["wilcoxon_on_protein_auc_diff"][name]:
            res["wilcoxon_on_protein_auc_diff"][name]["p_holm"] = float("nan")

    # —— AUPR：样本级配对 bootstrap + protein median bootstrap + Wilcoxon（各 3 组比较 + Holm）
    res["pairwise_sample_bootstrap_aupr"] = {}
    raw_sa: list[float] = []
    labels_sa: list[str] = []
    for name, s1, s2, stat_label in pairs:
        r = bootstrap_aupr_diff_paired(
            y, s1, s2, n_boot=800, boot_size_cap=100_000, stat=stat_label
        )
        r["pairing"] = "paired_same_intersection_instance_three_methods"
        r["unit"] = "sample_level_aupr_delta"
        r["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["pairwise_sample_bootstrap_aupr"][name] = r
        raw_sa.append(r["p_value_two_sided_bootstrap"])
        labels_sa.append(name)
    adj_sa = holm_adjust(raw_sa)
    for name, ap in zip(labels_sa, adj_sa):
        res["pairwise_sample_bootstrap_aupr"][name]["p_holm"] = float(ap)

    df_i_a = per_protein_metric_from_merged(merged_ok, "pred_im", _aupr, "AUPR")
    df_n_a = per_protein_metric_from_merged(merged_ok, "pred_nm", _aupr, "AUPR")
    df_m_a = per_protein_metric_from_merged(merged_ok, "pred_mix", _aupr, "AUPR")
    res["protein_median_bootstrap_aupr"] = {}
    raw_pa: list[float] = []
    labels_pa: list[str] = []
    for name, a, b in [
        ("ImmuScope_vs_NetMHCIIpan", df_i_a, df_n_a),
        ("ImmuScope_vs_MixMHC2pred", df_i_a, df_m_a),
        ("NetMHCIIpan_vs_MixMHC2pred", df_n_a, df_m_a),
    ]:
        rp = bootstrap_median_protein_auc_diff(a, b, auc_col="AUPR", n_boot=5000)
        rp["pairing"] = "paired_per_protein_on_intersection_long_table"
        rp["unit"] = "protein_level_median_aupr_delta"
        rp["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["protein_median_bootstrap_aupr"][name] = rp
        raw_pa.append(rp["p_value_two_sided_bootstrap"])
        labels_pa.append(name)
    valid_pa = [i for i, p in enumerate(raw_pa) if np.isfinite(p)]
    if valid_pa:
        sub_p = [raw_pa[i] for i in valid_pa]
        sub_adj = holm_adjust(sub_p)
        j = 0
        for i in range(len(labels_pa)):
            if i in valid_pa:
                res["protein_median_bootstrap_aupr"][labels_pa[i]]["p_holm"] = float(sub_adj[j])
                j += 1
            else:
                res["protein_median_bootstrap_aupr"][labels_pa[i]]["p_holm"] = float("nan")

    prot_pairs_aupr = [
        ("ImmuScope_vs_NetMHCIIpan", df_i_a, df_n_a),
        ("ImmuScope_vs_MixMHC2pred", df_i_a, df_m_a),
        ("NetMHCIIpan_vs_MixMHC2pred", df_n_a, df_m_a),
    ]
    res["wilcoxon_on_protein_aupr_diff"] = {}
    raw_wa: list[float] = []
    names_wa: list[str] = []
    for name, a, b in prot_pairs_aupr:
        wr = wilcoxon_protein_auc_diff(a, b, auc_col="AUPR")
        wr["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["wilcoxon_on_protein_aupr_diff"][name] = wr
        pv = wr.get("p_value_two_sided", float("nan"))
        if "error" not in wr and np.isfinite(pv):
            raw_wa.append(float(pv))
            names_wa.append(name)
    if raw_wa:
        adj_wa = holm_adjust(raw_wa)
        for nm, ap in zip(names_wa, adj_wa):
            res["wilcoxon_on_protein_aupr_diff"][nm]["p_holm"] = float(ap)
    for name, _, _ in prot_pairs_aupr:
        if "p_holm" not in res["wilcoxon_on_protein_aupr_diff"][name]:
            res["wilcoxon_on_protein_aupr_diff"][name]["p_holm"] = float("nan")

    # —— PPV（阈值 0.5）：同上
    res["pairwise_sample_bootstrap_ppv"] = {}
    raw_sp: list[float] = []
    labels_sp: list[str] = []
    for name, s1, s2, stat_label in pairs:
        r = bootstrap_ppv_diff_paired(
            y, s1, s2, n_boot=800, boot_size_cap=100_000, stat=stat_label
        )
        r["pairing"] = "paired_same_intersection_instance_three_methods"
        r["unit"] = "sample_level_ppv_delta"
        r["multiple_comparison"] = "holm_across_3_pairwise_tests"
        res["pairwise_sample_bootstrap_ppv"][name] = r
        raw_sp.append(r["p_value_two_sided_bootstrap"])
        labels_sp.append(name)
    adj_sp = holm_adjust(raw_sp)
    for name, ap in zip(labels_sp, adj_sp):
        res["pairwise_sample_bootstrap_ppv"][name]["p_holm"] = float(ap)

    df_i_p = per_protein_metric_from_merged(merged_ok, "pred_im", _ppv, "PPV")
    df_n_p = per_protein_metric_from_merged(merged_ok, "pred_nm", _ppv, "PPV")
    df_m_p = per_protein_metric_from_merged(merged_ok, "pred_mix", _ppv, "PPV")
    res["protein_median_bootstrap_ppv"] = {}
    raw_pp: list[float] = []
    labels_pp: list[str] = []
    for name, a, b in [
        ("ImmuScope_vs_NetMHCIIpan", df_i_p, df_n_p),
        ("ImmuScope_vs_MixMHC2pred", df_i_p, df_m_p),
        ("NetMHCIIpan_vs_MixMHC2pred", df_n_p, df_m_p),
    ]:
        rp = bootstrap_median_protein_auc_diff(a, b, auc_col="PPV", n_boot=5000)
        rp["pairing"] = "paired_per_protein_on_intersection_long_table"
        rp["unit"] = "protein_level_median_ppv_delta"
        rp["multiple_comparison"] = "holm_across_3_pairwise_tests"
        rp["ppv_threshold"] = float(PPV_DEFAULT_THRESHOLD)
        res["protein_median_bootstrap_ppv"][name] = rp
        raw_pp.append(rp["p_value_two_sided_bootstrap"])
        labels_pp.append(name)
    valid_pp = [i for i, p in enumerate(raw_pp) if np.isfinite(p)]
    if valid_pp:
        sub_p = [raw_pp[i] for i in valid_pp]
        sub_adj = holm_adjust(sub_p)
        j = 0
        for i in range(len(labels_pp)):
            if i in valid_pp:
                res["protein_median_bootstrap_ppv"][labels_pp[i]]["p_holm"] = float(sub_adj[j])
                j += 1
            else:
                res["protein_median_bootstrap_ppv"][labels_pp[i]]["p_holm"] = float("nan")

    prot_pairs_ppv = [
        ("ImmuScope_vs_NetMHCIIpan", df_i_p, df_n_p),
        ("ImmuScope_vs_MixMHC2pred", df_i_p, df_m_p),
        ("NetMHCIIpan_vs_MixMHC2pred", df_n_p, df_m_p),
    ]
    res["wilcoxon_on_protein_ppv_diff"] = {}
    raw_wp: list[float] = []
    names_wp: list[str] = []
    for name, a, b in prot_pairs_ppv:
        wr = wilcoxon_protein_auc_diff(a, b, auc_col="PPV")
        wr["multiple_comparison"] = "holm_across_3_pairwise_tests"
        wr["ppv_threshold"] = float(PPV_DEFAULT_THRESHOLD)
        res["wilcoxon_on_protein_ppv_diff"][name] = wr
        pv = wr.get("p_value_two_sided", float("nan"))
        if "error" not in wr and np.isfinite(pv):
            raw_wp.append(float(pv))
            names_wp.append(name)
    if raw_wp:
        adj_wp = holm_adjust(raw_wp)
        for nm, ap in zip(names_wp, adj_wp):
            res["wilcoxon_on_protein_ppv_diff"][nm]["p_holm"] = float(ap)
    for name, _, _ in prot_pairs_ppv:
        if "p_holm" not in res["wilcoxon_on_protein_ppv_diff"][name]:
            res["wilcoxon_on_protein_ppv_diff"][name]["p_holm"] = float("nan")

    res["ppv_threshold"] = float(PPV_DEFAULT_THRESHOLD)

    cmp_path = proj / "results/compare_cd4_auc_three.csv"
    if cmp_path.is_file():
        try:
            tab = pd.read_csv(cmp_path)
            if "n_intersection_rows" in tab.columns:
                exp = int(tab["n_intersection_rows"].iloc[0])
                res["compare_csv_n_intersection_rows"] = exp
                res["intersection_rowcount_match"] = bool(exp == len(y))
        except Exception:  # noqa: BLE001
            pass
    return res


def motif_wilcoxon_paired_allele_jsd_diffs(df: pd.DataFrame) -> dict | None:
    """
    当存在 **allele** 与 **jsd_mean** 列时：对三种 motif 工具对在 **同一 allele** 上的 JSD 做差，
    再对差向量做 **双侧 Wilcoxon**（与 3.2 protein 层 Wilcoxon 同一 scipy 接口）。
    三组差分 + Holm；若某组 inner merge 后 allele < 3 则跳过该组。
    """
    required = {"allele", "jsd_mean", "method_a", "method_b"}
    if not required.issubset(df.columns):
        return None

    def _one(ma: str, mb: str) -> pd.DataFrame:
        sub = df[(df["method_a"] == ma) & (df["method_b"] == mb)][["allele", "jsd_mean"]].copy()
        return sub.rename(columns={"jsd_mean": "jsd"})

    imm_net = _one("ImmuScope", "NetMHCIIpan")
    imm_mix = _one("ImmuScope", "MixMHC2pred")
    net_mix = _one("NetMHCIIpan", "MixMHC2pred")

    contrast_defs: list[tuple[str, pd.DataFrame, pd.DataFrame, str]] = [
        (
            "jsd_ImmuScope_vs_Net_minus_jsd_ImmuScope_vs_Mix",
            imm_net,
            imm_mix,
            "Per allele: JSD(ImmuScope,NetMHCIIpan) − JSD(ImmuScope,MixMHC2pred).",
        ),
        (
            "jsd_ImmuScope_vs_Net_minus_jsd_Net_vs_Mix",
            imm_net,
            net_mix,
            "Per allele: JSD(ImmuScope,NetMHCIIpan) − JSD(NetMHCIIpan,MixMHC2pred).",
        ),
        (
            "jsd_ImmuScope_vs_Mix_minus_jsd_Net_vs_Mix",
            imm_mix,
            net_mix,
            "Per allele: JSD(ImmuScope,MixMHC2pred) − JSD(NetMHCIIpan,MixMHC2pred).",
        ),
    ]

    out: dict = {
        "protocol": (
            "Optional Wilcoxon when per-allele jsd_mean rows exist: inner-merge on allele, "
            "delta JSD between two method-pair summaries, signed-rank test vs 0; Holm across 3 contrasts."
        ),
        "contrasts": {},
    }
    raw_p: list[float] = []
    labels_ok: list[str] = []

    for key, left, right, note in contrast_defs:
        row: dict = {"note": note}
        if len(left) == 0 or len(right) == 0:
            row["error"] = "missing_rows_for_one_or_both_method_pairs"
            out["contrasts"][key] = row
            continue
        mg = left.merge(right, on="allele", suffixes=("_L", "_R"), how="inner")
        d_arr = mg["jsd_L"].to_numpy(dtype=float) - mg["jsd_R"].to_numpy(dtype=float)
        n_alleles = int(len(d_arr))
        row["n_alleles"] = n_alleles
        if n_alleles < 3:
            row["error"] = "too_few_alleles_after_merge_for_wilcoxon"
            out["contrasts"][key] = row
            continue
        row["pairing"] = "paired_on_allele_after_inner_merge"
        row["unit"] = "delta_jsd_between_method_pair_summaries"
        if np.allclose(d_arr, 0):
            row["median_delta_jsd"] = 0.0
            row["wilcoxon_statistic"] = 0.0
            row["p_value_two_sided"] = 1.0
            out["contrasts"][key] = row
            raw_p.append(1.0)
            labels_ok.append(key)
            continue
        w_stat, pv = stats.wilcoxon(d_arr, zero_method="wilcox", alternative="two-sided")
        row["median_delta_jsd"] = float(np.median(d_arr))
        row["wilcoxon_statistic"] = float(w_stat)
        row["p_value_two_sided"] = float(pv)
        out["contrasts"][key] = row
        raw_p.append(float(pv))
        labels_ok.append(key)

    if labels_ok:
        adj = holm_adjust(raw_p)
        for k, ap in zip(labels_ok, adj):
            out["contrasts"][k]["p_holm"] = float(ap)
            out["contrasts"][k]["multiple_comparison"] = "holm_across_contrasts_with_valid_wilcoxon"
    return out


def section_3_3(proj: Path, out: Path) -> dict:
    """3.3 motif：按 allele 有放回 bootstrap 平均 JSD 的置信区间（非 p 值为主）。"""
    path = proj / "results/motif_compare/motif_pairwise_similarity.csv"
    res = {"section": "3.3"}
    if not path.is_file():
        res["error"] = f"missing {path}"
        return res
    df = pd.read_csv(path)
    rng = np.random.default_rng(42)
    B = 5000
    pairs = df[["method_a", "method_b"]].drop_duplicates()
    boot_summary = {}
    for _, row in pairs.iterrows():
        ma, mb = row["method_a"], row["method_b"]
        sub = df[(df["method_a"] == ma) & (df["method_b"] == mb)]
        alleles = sub["allele"].tolist()
        vals = sub["jsd_mean"].to_numpy(dtype=float)
        n = len(vals)
        obs = float(np.mean(vals))
        means = np.empty(B, dtype=float)
        aid = np.arange(n)
        for b in range(B):
            j = rng.choice(aid, size=n, replace=True)
            means[b] = float(np.mean(vals[j]))
        ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
        key = f"{ma}__vs__{mb}"
        boot_summary[key] = {
            "n_alleles": n,
            "observed_mean_jsd": obs,
            "bootstrap_B": B,
            "ci95_mean_jsd": [float(ci_lo), float(ci_hi)],
            "pairing": "cluster_bootstrap_by_allele",
            "unit": "allele_level_jsd_then_mean",
            "interpretation": "CI reflects allele sampling; not a test vs zero divergence.",
        }
    res["motif_allele_bootstrap"] = boot_summary

    wx = motif_wilcoxon_paired_allele_jsd_diffs(df)
    if wx is not None:
        res["wilcoxon_paired_allele_jsd_diff"] = wx

    res["aupr_ppv"] = {
        "applicable": False,
        "note_en": (
            "Section 3.3 summarizes **motif similarity (JSD)** per allele, not instance-level class labels "
            "and scores. **AUPR** and **threshold-based PPV** require binary labels and ranked predictions "
            "on the same instances; they are reported in **3.1 / 3.2 / 3.4 (IM) / 3.5 / 3.6** where paired "
            "predictions exist."
        ),
        "note_zh": (
            "3.3 为 **motif / JSD** 的按 allele 汇总，不含可与 **AUPR、固定阈值 PPV** 对齐的实例级标签与分数。"
            "**AUPR / PPV** 需在成对预测与二分类标签上计算，见 **3.1、3.2、3.4（IM）、3.5、3.6**。"
        ),
    }

    return res


def _friedman_wilcoxon_posthoc(X: np.ndarray, condition_names: list[str]) -> tuple[dict, list[dict]]:
    """Friedman on (n_blocks × k) matrix; Holm-adjusted Wilcoxon on pairwise column diffs."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if k < 3 or n < 2:
        return (
            {
                "n_blocks": int(n),
                "k_conditions": int(k),
                "error": "need k>=3 and n>=2",
            },
            [],
        )
    cols = [X[:, i] for i in range(k)]
    stat_f, p_f = stats.friedmanchisquare(*cols)
    fried = {
        "n_blocks": int(n),
        "k_conditions": int(k),
        "condition_labels": list(condition_names),
        "statistic_chi2": float(stat_f),
        "statistic": float(stat_f),
        "p_value": float(p_f),
        "n_repeats_training": 1,
        "limitation": (
            "Single checkpoint per condition; p reflects block-level (protein / allele / MHC) "
            "variation only, not retraining RNG."
        ),
    }
    pairs: list[str] = []
    raw_p: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            d = X[:, i] - X[:, j]
            lab = f"{condition_names[i]}_vs_{condition_names[j]}"
            if np.allclose(d, 0):
                pairs.append(lab)
                raw_p.append(1.0)
                continue
            _, pv = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            pairs.append(lab)
            raw_p.append(float(pv))
    adj = holm_adjust(raw_p)
    posthoc = [
        {"contrast": lab, "p_raw": rp, "p_holm": ap}
        for lab, rp, ap in zip(pairs, raw_p, adj, strict=True)
    ]
    return fried, posthoc


def _join_cd4_protein_aucs(
    paths: list[tuple[str, Path]],
) -> tuple[np.ndarray | None, list[str], dict[str, str]]:
    """内连接各 CSV 的 Protein → AUC，返回 (n_protein × k) 矩阵与列名。"""
    path_map = {n: str(p.resolve()) for n, p in paths}
    dfs = []
    for name, p in paths:
        if not p.is_file():
            return None, [], path_map
        dfs.append(pd.read_csv(p).rename(columns={"AUC": name}).set_index("Protein"))
    mat = dfs[0]
    for d in dfs[1:]:
        mat = mat.join(d, how="inner")
    names = [n for n, _ in paths]
    cols = [c for c in names if c in mat.columns]
    if len(cols) != len(paths) or len(mat) < 3:
        return None, [], path_map
    X = mat[cols].to_numpy(dtype=float)
    return X, names, path_map


def _join_el_auc01(paths: list[tuple[str, Path]]) -> tuple[np.ndarray | None, list[str], dict[str, str]]:
    path_map = {n: str(p.resolve()) for n, p in paths}
    first_name, first_p = paths[0]
    if not first_p.is_file():
        return None, [], path_map
    mg = pd.read_csv(first_p)[["allele", "AUC0.1"]].rename(columns={"AUC0.1": first_name})
    for name, p in paths[1:]:
        if not p.is_file():
            return None, [], path_map
        t = pd.read_csv(p)[["allele", "AUC0.1"]].rename(columns={"AUC0.1": name})
        mg = mg.merge(t, on="allele", how="inner")
    names = [n for n, _ in paths]
    if len(mg) < 3:
        return None, [], path_map
    X = mg[names].to_numpy(dtype=float)
    return X, names, path_map


def _merge_im_predictions(paths: list[tuple[str, Path]], keys: list[str]) -> pd.DataFrame | None:
    """多份 IM avg 在 keys 上内连接，pred 列名为 pred_<name>。"""
    name0, p0 = paths[0]
    if not p0.is_file():
        return None
    df = pd.read_csv(p0)
    for c in keys:
        if c not in df.columns:
            return None
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.rename(columns={"pred": f"pred_{name0}"})
    use_cols = keys + [f"pred_{name0}"]
    df = df[use_cols].drop_duplicates(subset=keys, keep="first")
    for name, p in paths[1:]:
        if not p.is_file():
            return None
        t = pd.read_csv(p)
        t["label"] = pd.to_numeric(t["label"], errors="coerce")
        t = t.rename(columns={"pred": f"pred_{name}"})
        t = t[keys + [f"pred_{name}"]].drop_duplicates(subset=keys, keep="first")
        df = df.merge(t, on=keys, how="inner")
    return df


def _im_mhc_metric_matrix(
    df: pd.DataFrame,
    pred_cols: list[str],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray | None:
    """每个 MHC 一层：对 k 个 pred 列在同一 `y` 上算 metric，要求 k 个值均有限。"""
    rows: list[list[float]] = []
    for _, g in df.groupby("mhc", sort=False):
        y = g["label"].to_numpy(dtype=float)
        if len(np.unique(np.where(y > 0.5, 1, 0))) < 2:
            continue
        vec: list[float] = []
        ok = True
        for pc in pred_cols:
            s = pd.to_numeric(g[pc], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(s)):
                ok = False
                break
            v = metric_fn(y, s)
            if not np.isfinite(v):
                ok = False
                break
            vec.append(float(v))
        if ok:
            rows.append(vec)
    if len(rows) < 3:
        return None
    return np.asarray(rows, dtype=float)


def _im_mhc_auc_matrix(df: pd.DataFrame, pred_cols: list[str]) -> np.ndarray | None:
    return _im_mhc_metric_matrix(df, pred_cols, _auc)


def section_3_4(abl: Path, rep: Path, out: Path) -> dict:
    """
    3.4 消融 + **reproduction 基线（Full）**：
      - **CD4**：`results_auc_protein_avg.csv` 在 **Protein** 上内连接 **Full + A1–A4** → Friedman + 10 组 Wilcoxon + Holm。
      - **EL**：各 `results_*_EL*_avg.csv` 在 **allele** 上内连接 **AUC0.1** → 同上（k=5）。
      - **IM**：各 `results_*_IM*_avg.csv` 在 **(mhc, peptide, label)** 上内连接 → 按 **MHC** 分层算 5 个 AUC → Friedman + Holm。
    """
    res: dict = {
        "section": "3.4",
        "design": "Full (ImmuScope_reproduction EL cascade) + ablation A1–A4 (ImmuScope_ablation)",
        "pairing_summary": (
            "CD4: blocked by protein; EL: blocked by allele summary row; "
            "IM: blocked by MHC (AUC from same merged test rows per MHC)."
        ),
        "notes": [],
    }

    cd4_paths: list[tuple[str, Path]] = [
        ("Full", rep / "results/ImmuScope-CD4/results_auc_protein_avg.csv"),
        ("A1", abl / "A1/results/ImmuScope-CD4/results_auc_protein_avg.csv"),
        ("A2", abl / "A2/results/ImmuScope-CD4/results_auc_protein_avg.csv"),
        ("A3", abl / "A3/results/ImmuScope-CD4/results_auc_protein_avg.csv"),
        ("A4", abl / "A4/results/ImmuScope-CD4/results_auc_protein_avg.csv"),
    ]
    X_cd4, names_cd4, pmap_cd4 = _join_cd4_protein_aucs(cd4_paths)
    res["cd4"] = {"csv_paths": pmap_cd4}
    if X_cd4 is None:
        res["cd4"]["error"] = "missing_csv_or_empty_join"
    else:
        fr, ph = _friedman_wilcoxon_posthoc(X_cd4, names_cd4)
        fr["pairing"] = "blocked_same_proteins_across_Full_and_A1_to_A4"
        fr["unit"] = "protein_level_auc"
        res["cd4"]["friedman_protein_auc"] = fr
        res["cd4"]["posthoc_wilcoxon_signed_rank_protein_auc_diff"] = ph
        res["cd4"]["n_pairwise_tests"] = len(ph)
        res["cd4"]["holm_note"] = "Holm across all pairwise contrasts for CD4 protein AUC."
        res["cd4"]["aupr_ppv_note_en"] = (
            "CD4 ablation track uses `results_auc_protein_avg.csv` (**one AUC per protein**). "
            "**AUPR** and threshold-**PPV** need instance-level labels and scores inside each protein; "
            "they are **not** recomputed here. Use **3.4 IM** (merged per-instance preds) or re-export "
            "per-sample CD4 tables for PR metrics."
        )
        res["cd4"]["aupr_ppv_note_zh"] = (
            "CD4 消融使用 `results_auc_protein_avg.csv`（**每蛋白一条 AUC**），无实例级分数，"
            "此处**不**计算 AUPR/PPV；PR 类指标见 **3.4 IM** 或需 per-sample CD4 导出。"
        )

    el_paths: list[tuple[str, Path]] = [
        ("Full", rep / "results/ImmuScope-EL/results_ImmuScope-EL_avg.csv"),
        ("A1", abl / "A1/results/ImmuScope-EL-no-si/results_ImmuScope-EL-no-si_avg.csv"),
        ("A2", abl / "A2/results/ImmuScope-EL-no-si-no-ftb/results_ImmuScope-EL-no-si-no-ftb_avg.csv"),
        (
            "A3",
            abl
            / "A3/results/ImmuScope-EL-no-si-no-ftb-A3-supcon/results_ImmuScope-EL-no-si-no-ftb-A3-supcon_avg.csv",
        ),
        (
            "A4",
            abl
            / "A4/results/ImmuScope-EL-no-si-no-ftb-A4-no-metric/results_ImmuScope-EL-no-si-no-ftb-A4-no-metric_avg.csv",
        ),
    ]
    X_el, names_el, pmap_el = _join_el_auc01(el_paths)
    res["el"] = {"csv_paths": pmap_el}
    if X_el is None or (isinstance(X_el, np.ndarray) and X_el.size == 0):
        res["el"]["error"] = "missing_csv_or_empty_join"
    else:
        fr_e, ph_e = _friedman_wilcoxon_posthoc(X_el, names_el)
        fr_e["pairing"] = "blocked_same_allele_rows_across_Full_and_A1_to_A4"
        fr_e["unit"] = "allele_level_AUC0.1_summary_csv"
        fr_e["caveat"] = (
            "Each allele's AUC0.1 is an EL-track summary; if peptide sets differ across runs, "
            "this is exploratory (same caveat as EL comparisons elsewhere)."
        )
        res["el"]["friedman_allele_auc01"] = fr_e
        res["el"]["posthoc_wilcoxon_signed_rank_allele_auc01_diff"] = ph_e
        res["el"]["aupr_ppv_note_en"] = (
            "EL summaries provide **AUC0.1** per allele, not instance-level scores; **AUPR/PPV** are not defined "
            "on this table."
        )
        res["el"]["aupr_ppv_note_zh"] = (
            "EL 汇总表仅有 **AUC0.1**，无实例级分数，**无法**在此表上定义 AUPR/PPV。"
        )

    im_paths: list[tuple[str, Path]] = [
        ("Full", rep / "results/ImmuScope-IM/results_ImmuScope-IM_avg.csv"),
        ("A1", abl / "A1/results/ImmuScope-IM-from-EL-A1/results_ImmuScope-IM-from-EL-A1_avg.csv"),
        ("A2", abl / "A2/results/ImmuScope-IM-from-EL-A2/results_ImmuScope-IM-from-EL-A2_avg.csv"),
        ("A3", abl / "A3/results/ImmuScope-IM-from-EL-A3/results_ImmuScope-IM-from-EL-A3_avg.csv"),
        ("A4", abl / "A4/results/ImmuScope-IM-from-EL-A4/results_ImmuScope-IM-from-EL-A4_avg.csv"),
    ]
    res["im"] = {"csv_paths": {n: str(p.resolve()) for n, p in im_paths}}
    keys_im = ["mhc", "peptide", "label"]
    merged_im = _merge_im_predictions(im_paths, keys_im)
    if merged_im is None or len(merged_im) < 100:
        res["im"]["error"] = "missing_csv_or_empty_join"
    else:
        pred_cols = [f"pred_{n}" for n, _ in im_paths]
        X_im = _im_mhc_auc_matrix(merged_im, pred_cols)
        res["im"]["n_merged_instances"] = int(len(merged_im))
        if X_im is None:
            res["im"]["error"] = "insufficient_mhc_strata_for_friedman"
        else:
            names_im = [n for n, _ in im_paths]
            fr_i, ph_i = _friedman_wilcoxon_posthoc(X_im, names_im)
            fr_i["pairing"] = "blocked_by_MHC_same_merged_test_rows"
            fr_i["unit"] = "mhc_level_auc_from_paired_predictions"
            res["im"]["friedman_mhc_auc"] = fr_i
            res["im"]["posthoc_wilcoxon_signed_rank_mhc_auc_diff"] = ph_i
            res["im"]["n_mhc_blocks"] = int(X_im.shape[0])

            X_im_a = _im_mhc_metric_matrix(merged_im, pred_cols, _aupr)
            if X_im_a is not None:
                fr_ia, ph_ia = _friedman_wilcoxon_posthoc(X_im_a, names_im)
                fr_ia["pairing"] = "blocked_by_MHC_same_merged_test_rows"
                fr_ia["unit"] = "mhc_level_aupr_from_paired_predictions"
                res["im"]["friedman_mhc_aupr"] = fr_ia
                res["im"]["posthoc_wilcoxon_signed_rank_mhc_aupr_diff"] = ph_ia
                res["im"]["n_mhc_blocks_aupr"] = int(X_im_a.shape[0])

            X_im_p = _im_mhc_metric_matrix(
                merged_im, pred_cols, lambda y, s: _ppv(y, s, PPV_DEFAULT_THRESHOLD)
            )
            if X_im_p is not None:
                fr_ip, ph_ip = _friedman_wilcoxon_posthoc(X_im_p, names_im)
                fr_ip["pairing"] = "blocked_by_MHC_same_merged_test_rows"
                fr_ip["unit"] = f"mhc_level_ppv_at_{PPV_DEFAULT_THRESHOLD}_from_paired_predictions"
                fr_ip["ppv_threshold"] = float(PPV_DEFAULT_THRESHOLD)
                res["im"]["friedman_mhc_ppv"] = fr_ip
                res["im"]["posthoc_wilcoxon_signed_rank_mhc_ppv_diff"] = ph_ip
                res["im"]["n_mhc_blocks_ppv"] = int(X_im_p.shape[0])
                res["im"]["ppv_threshold"] = float(PPV_DEFAULT_THRESHOLD)

    # 向后兼容：主 CD4 Friedman 仍暴露顶层键供旧版 md 渲染（可选）
    if "friedman_protein_auc" in res.get("cd4", {}):
        res["friedman_cd4_protein_auc"] = res["cd4"]["friedman_protein_auc"]
        res["posthoc_wilcoxon_signed_rank_on_protein_auc_diff"] = res["cd4"][
            "posthoc_wilcoxon_signed_rank_protein_auc_diff"
        ]

    res["notes"].append(
        "Omnibus Friedman and pairwise Holm-adjusted Wilcoxon are conditional on one checkpoint per label; "
        "they do not replace multi-seed training experiments."
    )
    return res


def section_3_5(
    proj: Path,
    baseline_im_avg: Path,
    integrate_im_avg: Path,
    out: Path,
) -> dict:
    """
    3.5 数据扩展：同一测试实例配对，样本级 bootstrap ΔAUC；
    另：按 MHC 分组 AUC 的 Wilcoxon（若可从逐样本重建）。
    """
    res = {"section": "3.5"}
    if not baseline_im_avg.is_file() or not integrate_im_avg.is_file():
        res["error"] = "missing IM csv"
        return res
    a = pd.read_csv(baseline_im_avg)
    b = pd.read_csv(integrate_im_avg)
    keys = ["mhc", "peptide", "label"]
    m = a.merge(b, on=keys, suffixes=("_base", "_new"))
    res["sample_bootstrap"] = bootstrap_auc_diff_paired(
        m["label"].to_numpy(),
        m["pred_base"].to_numpy(),
        m["pred_new"].to_numpy(),
        n_boot=800,
        boot_size_cap=None,
    )
    res["sample_bootstrap"]["pairing"] = "paired_mhc_peptide_label"
    res["sample_bootstrap"]["unit"] = "sample_level_auc_delta"
    res["n_merged"] = int(len(m))

    sb_a = bootstrap_aupr_diff_paired(
        m["label"].to_numpy(),
        m["pred_base"].to_numpy(),
        m["pred_new"].to_numpy(),
        n_boot=800,
        boot_size_cap=None,
        stat="delta_aupr_base_minus_new",
    )
    sb_a["pairing"] = "paired_mhc_peptide_label"
    sb_a["unit"] = "sample_level_aupr_delta"
    sb_a["delta_definition"] = "AUPR(base) - AUPR(new); negative implies higher AUPR for integrated model"
    res["sample_bootstrap_aupr"] = sb_a

    sb_p = bootstrap_ppv_diff_paired(
        m["label"].to_numpy(),
        m["pred_base"].to_numpy(),
        m["pred_new"].to_numpy(),
        n_boot=800,
        boot_size_cap=None,
        stat="delta_ppv_base_minus_new",
    )
    sb_p["pairing"] = "paired_mhc_peptide_label"
    sb_p["unit"] = "sample_level_ppv_delta"
    sb_p["delta_definition"] = (
        f"PPV(base) - PPV(new) at threshold {PPV_DEFAULT_THRESHOLD}; "
        "negative implies higher PPV for integrated model"
    )
    res["sample_bootstrap_ppv"] = sb_p

    def mhc_delta_auc(sub: pd.DataFrame) -> float | None:
        y = sub["label"].to_numpy(dtype=float)
        if len(np.unique(np.where(y > 0.5, 1, 0))) < 2:
            return None
        a0 = _auc(y, sub["pred_base"].to_numpy(dtype=float))
        a1 = _auc(y, sub["pred_new"].to_numpy(dtype=float))
        return float(a1 - a0)

    def mhc_delta_aupr(sub: pd.DataFrame) -> float | None:
        y = sub["label"].to_numpy(dtype=float)
        if len(np.unique(np.where(y > 0.5, 1, 0))) < 2:
            return None
        a0 = _aupr(y, sub["pred_base"].to_numpy(dtype=float))
        a1 = _aupr(y, sub["pred_new"].to_numpy(dtype=float))
        if not (np.isfinite(a0) and np.isfinite(a1)):
            return None
        return float(a1 - a0)

    def mhc_delta_ppv(sub: pd.DataFrame) -> float | None:
        y = sub["label"].to_numpy(dtype=float)
        if len(np.unique(np.where(y > 0.5, 1, 0))) < 2:
            return None
        a0 = _ppv(y, sub["pred_base"].to_numpy(dtype=float))
        a1 = _ppv(y, sub["pred_new"].to_numpy(dtype=float))
        if not (np.isfinite(a0) and np.isfinite(a1)):
            return None
        return float(a1 - a0)

    deltas = []
    mhcs = []
    for mhc_name, idx in m.groupby("mhc").groups.items():
        sub = m.loc[idx]
        du = mhc_delta_auc(sub)
        if du is not None and np.isfinite(du):
            deltas.append(du)
            mhcs.append(mhc_name)
    if len(deltas) >= 10:
        # 检验「跨 MHC 的 ΔAUC 是否系统偏离 0」：单样本 Wilcoxon on deltas
        w, p = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
        res["mhc_level_delta_auc"] = {
            "n_mhc_with_valid_auc": len(deltas),
            "median_delta_auc_across_mhc": float(np.median(deltas)),
            "wilcoxon_statistic": float(w),
            "p_value_two_sided": float(p),
            "pairing": "each_mhc_paired_predictions_then_auc_diff",
            "unit": "mhc_stratum_level_delta_auc",
            "note": "One delta per MHC; tests whether improvements concentrate across alleles.",
        }

    deltas_a: list[float] = []
    for mhc_name, idx in m.groupby("mhc").groups.items():
        sub = m.loc[idx]
        du = mhc_delta_aupr(sub)
        if du is not None and np.isfinite(du):
            deltas_a.append(du)
    if len(deltas_a) >= 10:
        w, p = stats.wilcoxon(deltas_a, zero_method="wilcox", alternative="two-sided")
        res["mhc_level_delta_aupr"] = {
            "n_mhc_with_valid_aupr": len(deltas_a),
            "median_delta_aupr_across_mhc": float(np.median(deltas_a)),
            "wilcoxon_statistic": float(w),
            "p_value_two_sided": float(p),
            "pairing": "each_mhc_paired_predictions_then_aupr_diff_new_minus_base",
            "unit": "mhc_stratum_level_delta_aupr",
            "note": "ΔAUPR = AUPR(new) − AUPR(base) per MHC; sign matches mhc_level_delta_auc convention.",
        }

    deltas_p: list[float] = []
    for mhc_name, idx in m.groupby("mhc").groups.items():
        sub = m.loc[idx]
        du = mhc_delta_ppv(sub)
        if du is not None and np.isfinite(du):
            deltas_p.append(du)
    if len(deltas_p) >= 10:
        w, p = stats.wilcoxon(deltas_p, zero_method="wilcox", alternative="two-sided")
        res["mhc_level_delta_ppv"] = {
            "n_mhc_with_valid_ppv": len(deltas_p),
            "median_delta_ppv_across_mhc": float(np.median(deltas_p)),
            "wilcoxon_statistic": float(w),
            "p_value_two_sided": float(p),
            "ppv_threshold": float(PPV_DEFAULT_THRESHOLD),
            "pairing": "each_mhc_paired_predictions_then_ppv_diff_new_minus_base",
            "unit": "mhc_stratum_level_delta_ppv",
        }

    return res


def main() -> None:
    if roc_auc_score is None or average_precision_score is None:
        sys.exit("sklearn required (roc_auc_score, average_precision_score)")

    p = argparse.ArgumentParser()
    p.add_argument("--reproduction-dir", type=Path, default=ROOT.parent / "ImmuScope_reproduction")
    p.add_argument("--ablation-dir", type=Path, default=ROOT.parent / "ImmuScope_ablation")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument(
        "--baseline-im",
        type=Path,
        default=None,
        help="3.5 对照 IM 预测表（mhc,peptide,label,pred）",
    )
    p.add_argument(
        "--integrate-im",
        type=Path,
        default=ROOT / "results/ImmuScope_IM_integrate_new_data/results/ImmuScope-IM/results_ImmuScope-IM_avg.csv",
    )
    p.add_argument(
        "--partition-variant-root",
        type=Path,
        default=None,
        help=(
            "若指定（或默认可探测到 ../ImmuScope_change_division），写入 JSON 第 3.6 节："
            "数据划分变体 vs --reproduction-dir 的 CD4/IM/EL 推断"
        ),
    )
    args = p.parse_args()

    rep = args.reproduction_dir.resolve()
    abl = args.ablation_dir.resolve()
    proj = args.project_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    baseline_im = args.baseline_im or (
        rep / "results/ImmuScope-IM-originalweight/results_ImmuScope-IM_avg.csv"
    )
    baseline_im = baseline_im.resolve()
    integrate_im = args.integrate_im.resolve()

    results = {}
    results["3.1"] = section_3_1(rep, OUT)
    results["3.2"] = section_3_2(proj, OUT)
    results["3.3"] = section_3_3(proj, OUT)
    results["3.4"] = section_3_4(abl, rep, OUT)
    results["3.5"] = section_3_5(proj, baseline_im, integrate_im, OUT)

    part_root = args.partition_variant_root
    if part_root is None:
        _cand = ROOT.parent / "ImmuScope_change_division"
        _probe = _cand / "results/ImmuScope-CD4/results_pred_protein_avg.csv"
        part_root = _cand if _probe.is_file() else None
    else:
        part_root = part_root.resolve()
    if part_root is not None:
        _probe = part_root / "results/ImmuScope-CD4/results_pred_protein_avg.csv"
        if _probe.is_file():
            results["3.6"] = section_partition_vs_reproduction(part_root, rep)
        else:
            print(
                f"Warning: --partition-variant-root set but missing {_probe}; skip section 3.6",
                file=sys.stderr,
            )

    out_json = OUT / "inferential_stats_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    md_script = ROOT / "scripts/json_inferential_stats_to_readable.py"
    if md_script.is_file():
        r = subprocess.run(
            [sys.executable, str(md_script), "--json-path", str(out_json)],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print(
                "Warning: json_inferential_stats_to_readable.py failed; MD may be stale.",
                file=sys.stderr,
            )

    print(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()

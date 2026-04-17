#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 ImmuScope / NetMHCIIpan / MixMHC2pred 在 CD4 基准上的 AUC（与 main_cd4_epitope_test 相同协议）：
  - Median AUC：各 protein 子集 AUC 的中位数
  - Mean AUC：各 protein 子集 AUC 的均值
  - Overall AUC：全样本 ROC-AUC

模式：
  --mode intersection（默认）
      三份 results_pred_per_sample.csv 按 (mhc_names, peptide, accession, dup_id) 内连接，
      只在三种方法都有的样本上算 AUC（解决 MixMHC2pred 少行、行序不一致等问题）。
      accession：protein 字符串按 "|" 分割后的第二段（UniProt id）；dup_id：同键内第几条重复窗。

  --mode full
      各 CSV 单独算 AUC（行数可不同，仅提示 WARN）。

用法：

  python scripts/compare_cd4_auc_three.py
  python scripts/compare_cd4_auc_three.py --mode full
  python scripts/compare_cd4_auc_three.py --out-intersection-csv results/cd4_intersection_three.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ImmuScope.utils.utils import calculate_auc_base_protein


def accession_from_protein(s: str) -> str:
    """与 MixMHC2pred / ImmuScope protein 列兼容：取 UniProt 风格第二段。"""
    segs = str(s).split("|")
    if len(segs) >= 3:
        return segs[1]
    return str(s)[:40]


def load_pred_label_protein(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(path)
    if "pred" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path}: 需要列 pred, label")
    if "protein" in df.columns:
        protein_col = "protein"
    elif "protein_id" in df.columns:
        protein_col = "protein_id"
    else:
        raise ValueError(f"{path}: 需要 protein 或 protein_id")
    protein_ids = df[protein_col].astype(str).to_numpy()
    pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy(dtype=np.float64)
    labels = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(labels)
    n_drop = int((~mask).sum())
    protein_ids = protein_ids[mask]
    pred = pred[mask]
    labels = labels[mask]
    return protein_ids, pred, labels, n_drop


def add_acc_dup_id_im_nm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["acc"] = out["protein"].map(accession_from_protein)
    out["dup_id"] = out.groupby(["mhc_names", "peptide", "acc"]).cumcount()
    return out


def add_acc_dup_id_mix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["acc"] = out["protein_id"].map(accession_from_protein)
    out["dup_id"] = out.groupby(["mhc_names", "peptide", "acc"]).cumcount()
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Compare CD4 AUC for three methods")
    p.add_argument(
        "--immuscope-csv",
        type=Path,
        default=root / "results/ImmuScope-CD4/results_pred_per_sample.csv",
    )
    p.add_argument(
        "--netmhc-csv",
        type=Path,
        default=root / "results/NetMHCIIpan-CD4/results_pred_per_sample.csv",
    )
    p.add_argument(
        "--mixmhc-csv",
        type=Path,
        default=root / "results/MixMHC2pred-CD4/results_pred_per_sample.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=root / "results/compare_cd4_auc_three.csv",
        help="汇总表输出路径",
    )
    p.add_argument(
        "--mode",
        choices=("intersection", "full"),
        default="intersection",
        help="intersection：三表内连接后同一样本算 AUC；full：各表单独算",
    )
    p.add_argument(
        "--out-intersection-csv",
        type=Path,
        default=None,
        help="可选：写出内连接后的子表（含 pred_im/pred_nm/pred_mix）",
    )
    args = p.parse_args()

    paths = [
        ("ImmuScope", args.immuscope_csv),
        ("NetMHCIIpan", args.netmhc_csv),
        ("MixMHC2pred", args.mixmhc_csv),
    ]

    if args.mode == "full":
        rows = []
        ns = []
        for name, csv_path in paths:
            csv_path = csv_path.resolve()
            if not csv_path.is_file():
                print(f"[SKIP] 未找到 {name}: {csv_path}")
                continue
            protein_ids, pred, labels, n_drop = load_pred_label_protein(csv_path)
            if n_drop:
                print(f"[WARN] {name}: 丢弃 {n_drop} 行无效 pred/label")
            median_auc, mean_auc, overall_auc, _ = calculate_auc_base_protein(
                protein_ids, pred, labels
            )
            ns.append(len(pred))
            rows.append(
                {
                    "model": name,
                    "n_rows": len(pred),
                    "median_auc_protein": median_auc,
                    "mean_auc_protein": mean_auc,
                    "overall_auc": overall_auc,
                    "csv": str(csv_path),
                    "mode": "full",
                }
            )
        if not rows:
            sys.exit("没有可用的 CSV，请检查路径。")
        out = pd.DataFrame(rows)
        if len(set(ns)) > 1:
            print(
                "[WARN] 三份 CSV 行数不一致："
                + ", ".join(f"{r['model']}={r['n_rows']}" for r in rows)
            )
    else:
        im_p, nm_p, mx_p = args.immuscope_csv, args.netmhc_csv, args.mixmhc_csv
        for label, path in [("ImmuScope", im_p), ("NetMHCIIpan", nm_p), ("MixMHC2pred", mx_p)]:
            if not path.resolve().is_file():
                sys.exit(f"缺少文件: {label} -> {path}")

        im = pd.read_csv(im_p.resolve())
        nm = pd.read_csv(nm_p.resolve())
        mx = pd.read_csv(mx_p.resolve())

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

        need = {"pred_im", "pred_nm", "pred_mix"}
        if not need.issubset(set(merged.columns)):
            sys.exit(
                f"内连接后列名异常，需 pred_im/pred_nm/pred_mix，当前: {merged.columns.tolist()}"
            )

        n_inter = len(merged)
        print(
            f"[OK] 三表交集样本数: {n_inter} "
            f"(ImmuScope {len(im)}, NetMHCIIpan {len(nm)}, MixMHC2pred {len(mx)})"
        )

        if args.out_intersection_csv:
            keep = [
                "mhc_names",
                "peptide",
                "protein",
                "label",
                "acc",
                "dup_id",
                "pred_im",
                "pred_nm",
                "pred_mix",
            ]
            extra = [c for c in merged.columns if c not in keep]
            out_i = merged[keep + [c for c in extra if c in ("raw_rank_pct", "protein_id", "mix_allele")]]
            args.out_intersection_csv.parent.mkdir(parents=True, exist_ok=True)
            out_i.to_csv(args.out_intersection_csv, index=False)
            print(f"[OK] 交集明细: {args.out_intersection_csv}")

        protein_ids = merged["protein"].astype(str).to_numpy()
        labels = pd.to_numeric(merged["label"], errors="coerce").to_numpy(dtype=np.float64)

        rows = []
        for col, name in [
            ("pred_im", "ImmuScope"),
            ("pred_nm", "NetMHCIIpan"),
            ("pred_mix", "MixMHC2pred"),
        ]:
            pred = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(pred) & np.isfinite(labels)
            median_auc, mean_auc, overall_auc, _ = calculate_auc_base_protein(
                protein_ids[mask], pred[mask], labels[mask]
            )
            rows.append(
                {
                    "model": name,
                    "n_rows": int(mask.sum()),
                    "median_auc_protein": median_auc,
                    "mean_auc_protein": mean_auc,
                    "overall_auc": overall_auc,
                    "csv": str(im_p if name == "ImmuScope" else nm_p if name == "NetMHCIIpan" else mx_p),
                    "mode": "intersection",
                    "n_intersection_rows": n_inter,
                }
            )
        out = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("\n=== CD4 AUC 对比（与 main_cd4_epitope_test 协议一致）===\n")
    disp = out[
        ["model", "n_rows", "median_auc_protein", "mean_auc_protein", "overall_auc"]
    ].copy()
    for c in ["median_auc_protein", "mean_auc_protein", "overall_auc"]:
        disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    print(disp.to_string(index=False))
    print(f"\n已保存: {args.out_csv}")


if __name__ == "__main__":
    main()

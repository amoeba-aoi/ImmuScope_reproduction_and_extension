#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CD4 baseline 的 results_pred_per_sample.csv 计算与 main_cd4_epitope_test.py 相同的指标：
  - Median / Mean AUC（按 protein 聚合后再对 protein 取中位/均值）
  - Overall AUC（全样本）
  - results_auc_protein_avg.csv：每个 protein 的 AUC
  - results_pred_protein_avg.csv：与测试脚本格式一致的逐样本表（protein, pred, label）

支持的列名：
  - protein 分组：优先 `protein`，否则 `protein_id`（MixMHC2pred 导出）
  - label: `label`
  - pred: `pred`

用法：
  python scripts/auc_from_cd4_per_sample.py \\
    --csv results/NetMHCIIpan-CD4/results_pred_per_sample.csv

  python scripts/auc_from_cd4_per_sample.py \\
    --csv results/MixMHC2pred-CD4/results_pred_per_sample.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 从项目根运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ImmuScope.utils.utils import calculate_auc_base_protein


def main() -> None:
    p = argparse.ArgumentParser(description="AUC from CD4 per-sample CSV (NetMHCIIpan / MixMHC2pred / etc.)")
    p.add_argument("--csv", type=Path, required=True, help="results_pred_per_sample.csv")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认与 csv 同目录",
    )
    args = p.parse_args()

    path = args.csv.resolve()
    if not path.is_file():
        sys.exit(f"文件不存在: {path}")

    df = pd.read_csv(path)
    if "pred" not in df.columns or "label" not in df.columns:
        sys.exit("CSV 至少需要列: pred, label")

    if "protein" in df.columns:
        protein_col = "protein"
    elif "protein_id" in df.columns:
        protein_col = "protein_id"
    else:
        sys.exit("CSV 需要 protein 或 protein_id 列作为分组键")

    protein_ids = df[protein_col].astype(str).to_numpy()
    pred = pd.to_numeric(df["pred"], errors="coerce").to_numpy(dtype=np.float64)
    labels = pd.to_numeric(df["label"], errors="coerce").to_numpy(dtype=np.float64)

    mask = np.isfinite(pred) & np.isfinite(labels)
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"[WARN] 丢弃 {n_drop} 行（pred/label 非数值或 NaN）")
    protein_ids = protein_ids[mask]
    pred = pred[mask]
    labels = labels[mask]

    median_auc, mean_auc, avg_auc, res_df = calculate_auc_base_protein(
        protein_ids, pred, labels
    )

    out_dir = args.out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        {"protein": protein_ids, "pred": pred, "label": labels}
    )
    df_out.to_csv(out_dir / "results_pred_protein_avg.csv", index=False)
    res_df.to_csv(out_dir / "results_auc_protein_avg.csv", index=False)

    print(
        f"Median AUC (protein): {median_auc:.4f} | "
        f"Mean AUC (protein): {mean_auc:.4f} | "
        f"Overall AUC: {avg_auc:.4f}"
    )
    print(f"Wrote: {out_dir / 'results_pred_protein_avg.csv'}")
    print(f"Wrote: {out_dir / 'results_auc_protein_avg.csv'}")


if __name__ == "__main__":
    main()

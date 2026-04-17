# -*- coding: utf-8 -*-
"""
CD4 基准：从 NetMHCIIpan_eval.h5 读入样本，调用 NetMHCIIpan-4.3，按与 main_cd4_epitope_test.py
相同的 protein-level AUC 协议输出 CSV。

依赖：本仓库 ImmuScope.utils；系统已安装 docker（推荐 Apple Silicon 使用 linux/amd64 镜像）
或本机已安装并可执行的 NetMHCIIpan 二进制（见 --backend local）。

用法示例（Docker，默认）：

  python run_netmhc_cd4_baseline.py \\
    --data-cnf configs/data.yaml \\
    --docker-image ghcr.io/macromnex/netmhc2pan_mcp:latest \\
    --platform linux/amd64

本机二进制示例：

  export NETMHCIIPAN_HOME=/path/to/netMHCIIpan-4.3
  python run_netmhc_cd4_baseline.py --backend local --netmhc-bin /path/to/netMHCIIpan-4.3/netMHCIIpan

小规模试跑：

  python run_netmhc_cd4_baseline.py --max-rows 200
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from ImmuScope.utils.data_utils import restore_peptide_sequences
from ImmuScope.utils.utils import calculate_auc_base_protein


def _parse_netmhciipan_table(stdout: str) -> List[Dict[str, Any]]:
    """解析 NetMHCIIpan 标准表格式输出（与 netmhc2pan_mcp scripts/lib/parsers 一致）。"""
    lines = stdout.splitlines()
    in_results = False
    rows: List[Dict[str, Any]] = []
    for line in lines:
        if line.startswith(" Pos ") and "MHC" in line and "Peptide" in line:
            in_results = True
            continue
        if not in_results or not line.strip() or line.strip().startswith("-"):
            continue
        if line.startswith("Number of"):
            break
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            rank = float(parts[12]) if parts[12] != "NA" else 100.0
            score = float(parts[11]) if len(parts) > 11 and parts[11] != "NA" else None
            rows.append(
                {
                    "position": int(parts[0]),
                    "peptide": parts[2],
                    "score": score,
                    "rank": rank,
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def _rank_to_pred(rank: float, mode: str) -> float:
    """将 %Rank 转为与「越高越像阳性/强结合」一致的分值，供 ROC-AUC 使用。"""
    if mode == "one_minus_rank":
        return float(np.clip(1.0 - rank / 100.0, 0.0, 1.0))
    if mode == "neg_log10_rank":
        # rank 为百分位；避免 log(0)
        r = max(rank, 1e-6)
        return float(-np.log10(r / 100.0))
    raise ValueError(f"Unknown score mode: {mode}")


def _run_netmhc_local(
    netmhc_bin: Path,
    peptide_file: Path,
    allele: str,
) -> str:
    cmd = [
        str(netmhc_bin),
        "-inptype",
        "1",
        "-f",
        str(peptide_file),
        "-a",
        allele,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"NetMHCIIpan failed (allele={allele}): {r.stderr or r.stdout}"
        )
    return r.stdout


def _run_netmhc_docker(
    peptide_file: Path,
    allele: str,
    docker_image: str,
    platform: str,
    script_in_image: str,
) -> str:
    peptide_file = peptide_file.resolve()
    workdir = str(peptide_file.parent)
    in_name = peptide_file.name
    out_name = peptide_file.with_suffix(".netmhc_out.txt").name
    out_host = peptide_file.with_suffix(".netmhc_out.txt")

    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        platform,
        "-v",
        f"{workdir}:{workdir}",
        "-w",
        workdir,
        docker_image,
        "python",
        script_in_image,
        "--input",
        f"{workdir}/{in_name}",
        "--allele",
        allele,
        "--output",
        f"{workdir}/{out_name}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Docker NetMHCIIpan failed (allele={allele}): {r.stderr or r.stdout}"
        )
    if not out_host.is_file():
        raise FileNotFoundError(f"Expected output file missing: {out_host}")
    return out_host.read_text(encoding="utf-8", errors="replace")


def _align_predictions(
    expected: List[str],
    parsed: List[Dict[str, Any]],
) -> np.ndarray:
    """将 NetMHCIIpan 解析结果与输入肽顺序对齐；必要时按 peptide 字段匹配。"""
    if len(parsed) == len(expected):
        return np.array([p["rank"] for p in parsed], dtype=np.float64)

    by_pep: Dict[str, List[float]] = {}
    for p in parsed:
        seq = p["peptide"]
        by_pep.setdefault(seq, []).append(float(p["rank"]))

    ranks: List[float] = []
    for pep in expected:
        if pep not in by_pep or not by_pep[pep]:
            ranks.append(100.0)
        else:
            ranks.append(by_pep[pep].pop(0))
    return np.array(ranks, dtype=np.float64)


def load_h5_arrays(
    h5_path: Path,
    max_rows: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        n = len(f["labels"])
        if max_rows is not None:
            n = min(n, max_rows)
        emb = f["peptide_embedding"][:n]
        mhc = f["mhc_names"][:n]
        ctx = f["peptide_contexts"][:n]
        labels = np.asarray(f["labels"][:n], dtype=np.float32)
    mhc = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in mhc])
    ctx = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ctx])
    return emb, mhc, ctx, labels


def main() -> None:
    p = argparse.ArgumentParser(description="NetMHCIIpan baseline on CD4 h5 benchmark")
    p.add_argument(
        "-d",
        "--data-cnf",
        type=Path,
        default=Path("configs/data.yaml"),
        help="读取其中 test 与 results 路径（默认 configs/data.yaml）",
    )
    p.add_argument(
        "--h5",
        type=Path,
        default=None,
        help="直接指定 h5；若缺省则用 data-cnf 中的 test",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认 <results>/NetMHCIIpan-CD4（results 来自 data-cnf）",
    )
    p.add_argument(
        "--backend",
        choices=("docker", "local"),
        default="docker",
        help="docker：在容器内调用 peptide_prediction.py；local：直接调用 netMHCIIpan 可执行文件",
    )
    p.add_argument(
        "--docker-image",
        default="ghcr.io/macromnex/netmhc2pan_mcp:latest",
        help="--backend docker 时使用的镜像",
    )
    p.add_argument(
        "--platform",
        default="linux/amd64",
        help="Apple Silicon 上建议 linux/amd64",
    )
    p.add_argument(
        "--script-in-image",
        default="/app/scripts/peptide_prediction.py",
        help="镜像内 peptide_prediction.py 路径（netmhc2pan_mcp 默认）",
    )
    p.add_argument(
        "--netmhc-bin",
        type=Path,
        default=None,
        help="--backend local 时 netMHCIIpan 可执行文件路径；也可设环境变量 NETMHCIIPAN_BIN",
    )
    p.add_argument(
        "--score-mode",
        choices=("one_minus_rank", "neg_log10_rank"),
        default="one_minus_rank",
        help="将 %%Rank 转为 AUC 用分数：默认 1-rank/100（越大结合越强）",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="只跑前 N 条，用于试跑",
    )
    p.add_argument(
        "--save-per-sample",
        action="store_true",
        help="额外写出 results_pred_per_sample.csv（调试/核对）",
    )
    args = p.parse_args()

    yaml = YAML(typ="safe")
    data_cnf = yaml.load(args.data_cnf)
    h5_path = Path(args.h5) if args.h5 else Path(data_cnf["test"])
    if not h5_path.is_file():
        sys.exit(f"找不到 h5 文件: {h5_path.resolve()}")

    results_root = Path(data_cnf.get("results", "results"))
    out_dir = args.out_dir or (results_root / "NetMHCIIpan-CD4")
    out_dir.mkdir(parents=True, exist_ok=True)

    emb, mhc_names, protein_ids, labels = load_h5_arrays(h5_path, args.max_rows)
    peptides = restore_peptide_sequences(emb, peptide_pad=3)

    # 按等位基因分批调用 NetMHCIIpan，减少进程次数
    uniq_alleles = sorted(set(mhc_names.tolist()))
    rank_vec = np.full(len(peptides), np.nan, dtype=np.float64)

    tmp_root = Path(tempfile.mkdtemp(prefix="netmhc_cd4_"))
    for allele in uniq_alleles:
        idx = np.where(mhc_names == allele)[0]
        batch_peptides = [peptides[i] for i in idx]
        # 跳过空肽；NetMHCIIpan 对非法字符可能报错
        safe_list: List[str] = []
        safe_idx: List[int] = []
        for j, pep in zip(idx.tolist(), batch_peptides):
            if not pep or not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", pep):
                rank_vec[j] = 100.0
                continue
            safe_list.append(pep)
            safe_idx.append(j)
        if not safe_list:
            continue

        safe_allele = re.sub(r"[^A-Za-z0-9._-]+", "_", allele)
        peptide_file = tmp_root / f"peptides_{safe_allele}.txt"
        peptide_file.write_text("\n".join(safe_list) + "\n", encoding="utf-8")

        if args.backend == "local":
            netmhc_bin = args.netmhc_bin or os.environ.get("NETMHCIIPAN_BIN")
            if not netmhc_bin:
                sys.exit("local 模式需要 --netmhc-bin 或环境变量 NETMHCIIPAN_BIN")
            raw = _run_netmhc_local(Path(netmhc_bin), peptide_file, allele)
        else:
            raw = _run_netmhc_docker(
                peptide_file,
                allele,
                args.docker_image,
                args.platform,
                args.script_in_image,
            )

        parsed = _parse_netmhciipan_table(raw)
        ranks = _align_predictions(safe_list, parsed)
        for j, r in zip(safe_idx, ranks):
            rank_vec[j] = r

    # 未写入的 NaN 视为最差结合
    rank_vec = np.where(np.isnan(rank_vec), 100.0, rank_vec)
    pred = np.array([_rank_to_pred(float(r), args.score_mode) for r in rank_vec])

    df_pred = pd.DataFrame(
        {"protein": protein_ids, "pred": pred, "label": labels.astype(np.float64)}
    )
    median_auc, mean_auc, avg_auc, res_df = calculate_auc_base_protein(
        protein_ids, pred, labels
    )

    df_pred.to_csv(out_dir / "results_pred_protein_avg.csv", index=False)
    res_df.to_csv(out_dir / "results_auc_protein_avg.csv", index=False)

    if args.save_per_sample:
        pd.DataFrame(
            {
                "mhc_names": mhc_names,
                "peptide": peptides,
                "protein": protein_ids,
                "label": labels,
                "pred": pred,
                "raw_rank_pct": rank_vec,
            }
        ).to_csv(out_dir / "results_pred_per_sample.csv", index=False)

    print(
        f"Done. Median AUC: {median_auc:.4f}; Mean AUC: {mean_auc:.4f}; Overall AUC: {avg_auc:.4f}"
    )
    print(f"Wrote: {out_dir / 'results_pred_protein_avg.csv'}")
    print(f"Wrote: {out_dir / 'results_auc_protein_avg.csv'}")
    if args.save_per_sample:
        print(f"Wrote: {out_dir / 'results_pred_per_sample.csv'}")
    print(f"Temp dir (peptide batches): {tmp_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export ImmuScope CD4 per-sample predictions for motif comparison.

This script follows main_cd4_epitope_test.py prediction flow, and writes:
  results/ImmuScope-CD4/results_pred_per_sample.csv

Required output columns (for motif_deconv_compare.py):
  mhc_names, peptide, protein, label, pred
"""

import os
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ImmuScope.datasets.datasets import SinInstanceBag
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.models.trainer_cd4_epitope import Trainer
from ImmuScope.utils.data_utils import get_mhc_name_seq, restore_peptide_sequences
from ImmuScope.utils.utils import load_config_and_setup_logging


def _enable_cpu_checkpoint_loading_if_needed() -> None:
    """
    Trainer internally calls torch.load(path) without map_location.
    If CUDA is unavailable, patch torch.load to map checkpoints to CPU.
    """
    if torch.cuda.is_available():
        return
    original_torch_load = torch.load

    def _cpu_safe_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = _cpu_safe_torch_load


@torch.no_grad()
def _predict_with_batch_progress(
    trainer: Trainer,
    data_loader: DataLoader,
    model_id: int,
    logger,
    log_every_n_batches: int = 0,
):
    """
    Predict one model with batch-level tqdm progress.
    Returns numpy array of instance probabilities.
    """
    trainer.model.eval()
    pred_instance = []
    total_batches = len(data_loader)
    pbar = tqdm(
        data_loader,
        total=total_batches,
        desc=f"Model {model_id} batches",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    t0 = time.time()
    seen = 0
    for batch_idx, (inputs, labels) in enumerate(pbar, start=1):
        labels = labels.to(trainer.device)
        inputs = tuple(x.to(trainer.device) for x in inputs)
        _, instance_prob, _, _, _ = trainer.model(inputs)
        pred_instance.append(instance_prob.cpu().numpy())
        seen += len(labels)

        elapsed = time.time() - t0
        speed = seen / elapsed if elapsed > 0 else 0.0
        remain_batches = total_batches - batch_idx
        avg_batch_sec = elapsed / batch_idx
        eta_sec = remain_batches * avg_batch_sec
        pbar.set_postfix(
            {
                "seen": seen,
                "samples/s": f"{speed:.1f}",
                "ETA": f"{eta_sec/60:.1f}m",
            }
        )
        if log_every_n_batches > 0 and (
            batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
        ):
            logger.info(
                "Model %s batch %d/%d | seen=%d | %.1f samples/s | ETA %.1f min",
                model_id,
                batch_idx,
                total_batches,
                seen,
                speed,
                eta_sec / 60.0,
            )
    pbar.close()
    return np.hstack(pred_instance)


@click.command()
@click.option("-d", "--data-cnf", type=click.Path(exists=True), default="configs/data.yaml")
@click.option("-m", "--model-cnf", type=click.Path(exists=True), default="configs/ImmuScope.yaml")
@click.option("-s", "--start-id", default=0, show_default=True, type=int)
@click.option("-n", "--num-models", default=10, show_default=True, type=int)
@click.option(
    "--log-every-n-batches",
    default=0,
    show_default=True,
    type=int,
    help="If >0, write batch progress to log every N batches.",
)
def main(data_cnf, model_cnf, start_id, num_models, log_every_n_batches):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _enable_cpu_checkpoint_loading_if_needed()
    logger, data_cnf, model_cnf = load_config_and_setup_logging(
        data_cnf=data_cnf, model_cnf=model_cnf, logger_name="ImmuScope-CD4-PerSample"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_cnf["name"]
    all_model_path = Path(model_cnf["path"]) / "CD4" / f"{model_name}.pt"

    mhc_name_seq = get_mhc_name_seq(data_cnf["mhc_seq"])
    test_loader = DataLoader(
        SinInstanceBag(data_cnf["test"], mhc_name_seq, indices=None),
        batch_size=model_cnf["test"]["batch_size"],
    )
    total_samples = len(test_loader.dataset)
    model_ids = list(range(start_id, start_id + num_models))
    logger.info(
        "Start CD4 per-sample export | models=%d | samples/model=%d | device=%s",
        len(model_ids),
        total_samples,
        device,
    )

    pred_instances_models = []
    all_start = time.time()
    pbar = tqdm(model_ids, desc="CD4 test progress", unit="model", dynamic_ncols=True)
    for done_idx, model_id in enumerate(pbar, start=1):
        saved_model_path = all_model_path.with_stem(f"{all_model_path.stem}-{model_id}")
        trainer = Trainer(
            ImmuScope,
            model_path=saved_model_path,
            device=device,
            logger=logger,
            **model_cnf["model"],
        )
        model_start = time.time()
        trainer.load_model(saved_model_path)
        pred_instances = _predict_with_batch_progress(
            trainer,
            test_loader,
            model_id=model_id,
            logger=logger,
            log_every_n_batches=log_every_n_batches,
        )
        model_sec = time.time() - model_start
        pred_instances_models.append(pred_instances)
        speed = len(pred_instances) / model_sec if model_sec > 0 else 0.0

        elapsed = time.time() - all_start
        avg_per_model = elapsed / done_idx
        remain_models = len(model_ids) - done_idx
        eta_sec = remain_models * avg_per_model
        pbar.set_postfix(
            {
                "model": model_id,
                "sec/model": f"{model_sec:.1f}",
                "samples/s": f"{speed:.1f}",
                "ETA": f"{eta_sec/60:.1f}m",
            }
        )
        logger.info(
            "Model %s predicted %d samples | %.2fs | %.1f samples/s | ETA %.1f min",
            model_id,
            len(pred_instances),
            model_sec,
            speed,
            eta_sec / 60.0,
        )
    pbar.close()

    total_sec = time.time() - all_start
    logger.info(
        "Prediction finished | total %.2f min | avg %.2f sec/model",
        total_sec / 60.0,
        total_sec / max(len(model_ids), 1),
    )

    pred_mean = np.mean(np.array(pred_instances_models), axis=0)

    ds = test_loader.dataset
    mhc_names = ds.mhc_names
    proteins = ds.peptide_contexts
    labels = ds.labels
    peptides = restore_peptide_sequences(ds.peptide_embedding[:, 0, :], peptide_pad=3)

    df = pd.DataFrame(
        {
            "mhc_names": mhc_names,
            "peptide": peptides,
            "protein": proteins,
            "label": labels.astype(np.float64),
            "pred": pred_mean.astype(np.float64),
        }
    )

    out_dir = Path(data_cnf["results"]) / "ImmuScope-CD4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results_pred_per_sample.csv"
    df.to_csv(out_file, index=False)
    logger.info("Saved per-sample predictions: %s (rows=%d)", out_file, len(df))
    print(f"[OK] Wrote: {out_file}")


if __name__ == "__main__":
    main()


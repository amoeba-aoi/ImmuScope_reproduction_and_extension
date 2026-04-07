# -*- coding: utf-8 -*-
# 2026/4/5
# dongyizhe
"""
@Time ： 2024/10/12
@Auth ： shenlongchen
@Description : ImmuScope: Fine-tune from ImmuScope-EL with BA data.

Extensions (vs main_cd4_epitope_train.py):
  --el-stem / --el-suffix: point to any EL ablation checkpoint under weights/EL/
  --weights-tag: avoid overwriting CD4 weights when comparing EL variants
EL path helpers are inlined in this file (no separate utils module).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from typing import Optional

import click
from torch.utils.data.dataloader import DataLoader
from ImmuScope.datasets.datasets import SinInstanceBag
from ImmuScope.models.trainer_cd4_epitope import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *


def _el_checkpoint_path(weights_root: Path, el_stem: str, model_id: int, el_suffix: str) -> Path:
    """weights/EL/{el_stem}-{model_id}-{el_suffix}.pt"""
    el_dir = Path(weights_root) / "EL"
    base = el_dir / f"{el_stem}.pt"
    return base.with_stem(f"{base.stem}-{model_id}-{el_suffix}")


def _resolve_el_stem_cd4(model_name: str, el_stem: Optional[str]) -> str:
    if el_stem:
        return el_stem
    return f"{model_name}-EL"


def train_cd4_epitope_with_ba(trainer, pretrain_model, mhc_name_seq, data_cnf, model_cnf, logger, random_state=2024):
    logger.info(f'Start training model {trainer.model_path}')
    train_path_ba = data_cnf['train_ba']
    train_path_sa = data_cnf['train_sa']

    train_ba_idx, valid_ba_idx = create_splits(train_path_ba, split_ratio=0.1, seed=random_state)
    train_sa_idx, valid_sa_idx = create_splits(train_path_sa, split_ratio=0.1, seed=random_state)

    train_loader_ba = DataLoader(
        SinInstanceBag(train_path_ba, mhc_name_seq, indices=train_ba_idx), batch_size=model_cnf['train']['batch_size'],
        shuffle=True)
    train_loader_sa = DataLoader(
        SinInstanceBag(train_path_sa, mhc_name_seq, indices=train_sa_idx), batch_size=model_cnf['train']['batch_size'],
        shuffle=True)
    train_loader = [train_loader_ba, train_loader_sa]

    valid_loader_ba = DataLoader(
        SinInstanceBag(train_path_ba, mhc_name_seq, indices=valid_ba_idx), batch_size=model_cnf['valid']['batch_size'])
    valid_loader_sa = DataLoader(
        SinInstanceBag(train_path_sa, mhc_name_seq, indices=valid_sa_idx), batch_size=model_cnf['valid']['batch_size'])
    valid_loader = [valid_loader_ba, valid_loader_sa]

    test_loader = DataLoader(
        SinInstanceBag(data_cnf['test'], mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

    trainer.train_with_ba(train_loader, valid_loader, test_loader, pretrained_model_path=pretrain_model,
                          **model_cnf['train'])
    logger.info(f'Finish training model {trainer.model_path}')


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
@click.option(
    '--el-stem',
    default=None,
    help='EL checkpoint stem (no .pt / without -{id}-{suffix}). Default: {name}-EL from model yaml.',
)
@click.option(
    '--el-suffix',
    default='fine-tune-b',
    help='Suffix after model id in EL weights (fine-tune-b or pretrain).',
)
@click.option(
    '--weights-tag',
    default=None,
    help='Optional tag appended to CD4 save stem so EL variants do not overwrite each other.',
)
def main(data_cnf, model_cnf, start_id, num_models, el_stem, el_suffix, weights_tag):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]
    out_name = model_name if not weights_tag else f'{model_name}-{weights_tag}'

    el_stem_resolved = _resolve_el_stem_cd4(model_name, el_stem)
    weights_root = Path(model_cnf['path'])
    saved_cd4_peitope_path = Path(os.path.join(weights_root, 'CD4', f'{out_name}.pt'))
    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    pred_instances_models = []
    labels_models = []
    peptide_protein_models = []

    for model_id in range(start_id, start_id + num_models):
        pretrain_model = _el_checkpoint_path(weights_root, el_stem_resolved, model_id, el_suffix)
        if not pretrain_model.is_file():
            raise FileNotFoundError(
                f'Missing EL checkpoint for downstream CD4: {pretrain_model}\n'
                f'Expected: weights/EL/{{el_stem}}-{{id}}-{{el_suffix}}.pt (try --el-stem / --el-suffix).'
            )
        logger.info(f'Loading EL init from {pretrain_model}')
        saved_model_path = saved_cd4_peitope_path.with_stem(f'{saved_cd4_peitope_path.stem}-{model_id}')
        trainer = Trainer(ImmuScope, model_path=saved_model_path, device=device, logger=logger, **model_cnf['model'])

        train_cd4_epitope_with_ba(trainer, pretrain_model, mhc_name_seq, data_cnf, model_cnf, logger=logger)

        test_loader = DataLoader(
            SinInstanceBag(data_cnf['test'], mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

        pred_instances, _, _ = trainer.predict(test_loader, model_prefix="")
        peptide_protein = test_loader.dataset.peptide_contexts
        labels = test_loader.dataset.labels
        median_auc, mean_auc, avg_auc, res_df = calculate_auc_base_protein(peptide_protein, pred_instances, labels)
        logger.info('|**---------Model {}--- TEST: Median AUC: {:.4f}; Mean AUC: {:.4f}; AVG AUC: {:.4f}---------**|'.
                    format(model_id, median_auc, mean_auc, avg_auc))
        pred_instances_models.append(pred_instances)
        labels_models.append(labels)
        peptide_protein_models.append(peptide_protein)

    logger.info(f'-----------------Average-----------------')
    median_auc, mean_auc, avg_auc, res_df = calculate_auc_base_protein(peptide_protein_models[0],
                                                                       np.mean(np.array(pred_instances_models), axis=0),
                                                                       np.mean(np.array(labels_models), axis=0))
    logger.info('|**========== TEST: Median AUC: {:.4f}; Mean AUC: {:.4f}; AVG AUC: {:.4f}==========**|'.
                format(median_auc, mean_auc, avg_auc))


if __name__ == '__main__':
    main()

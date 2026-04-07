# -*- coding: utf-8 -*-
# 2026/4/5
# dongyizhe
"""
@Time ： 2024/2/29
@Auth ： shenlongchen
@Description : Fine-tune from ImmuScope-EL with immunogenicity data.

Extensions (vs main_immunogenicity_train.py):
  --el-stem / --el-suffix: any EL checkpoint; default el-stem matches original (ImmuScope-EL)
  --weights-tag: separate IM output dirs per EL variant
EL path helpers inlined below.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from typing import Optional

import click
from torch.utils.data.dataloader import DataLoader
from ImmuScope.datasets.datasets import SinInstanceBag
from ImmuScope.models.trainer_immunogenicity import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *


def _el_checkpoint_path(weights_root: Path, el_stem: str, model_id: int, el_suffix: str) -> Path:
    el_dir = Path(weights_root) / "EL"
    base = el_dir / f"{el_stem}.pt"
    return base.with_stem(f"{base.stem}-{model_id}-{el_suffix}")


def _resolve_el_stem_im(_model_name: str, el_stem: Optional[str]) -> str:
    """Default EL stem matches original main (fixed ImmuScope-EL, not IM yaml name)."""
    if el_stem:
        return el_stem
    return "ImmuScope-EL"


def train_immuscope_im(trainer, pretrain_model, mhc_name_seq, data_cnf, model_cnf, logger, random_state=2024):
    logger.info(f'Start training model {trainer.model_path}')
    train_path_imm = data_cnf['train_imm']
    test_path_imm = data_cnf['test_imm']

    train_ba_idx, valid_ba_idx = create_splits(train_path_imm, split_ratio=0.1, seed=random_state)

    train_loader = DataLoader(
        SinInstanceBag(train_path_imm, mhc_name_seq, indices=train_ba_idx), batch_size=model_cnf['train']['batch_size'],
        shuffle=True)

    valid_loader = DataLoader(
        SinInstanceBag(train_path_imm, mhc_name_seq, indices=valid_ba_idx), batch_size=model_cnf['valid']['batch_size'])

    test_loader = DataLoader(
        SinInstanceBag(test_path_imm, mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

    trainer.train_with_imm(train_loader, valid_loader, test_loader, pretrained_model_path=pretrain_model,
                           **model_cnf['train'])
    logger.info(f'Finish training model {trainer.model_path}')


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope-IM.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
@click.option('--el-stem', default=None, help='EL checkpoint stem. Default: ImmuScope-EL (original behavior).')
@click.option('--el-suffix', default='fine-tune-b', help='EL weight suffix after model id.')
@click.option('--weights-tag', default=None, help='Append to IM save/results name per EL variant.')
def main(data_cnf, model_cnf, start_id, num_models, el_stem, el_suffix, weights_tag):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]
    out_name = model_name if not weights_tag else f'{model_name}-{weights_tag}'
    result_path = Path.joinpath(Path(data_cnf['results']), out_name)
    result_path.mkdir(parents=True, exist_ok=True)

    el_stem_resolved = _resolve_el_stem_im(model_name, el_stem)
    weights_root = Path(model_cnf['path'])
    imm_model_path = Path(os.path.join(weights_root, 'IM', f'{out_name}.pt'))

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    pred_instances_models = []
    labels_models = []

    test_path_imm = data_cnf['test_imm']
    test_loader = DataLoader(
        SinInstanceBag(test_path_imm, mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

    for model_id in range(start_id, start_id + num_models):
        pretrain_model = _el_checkpoint_path(weights_root, el_stem_resolved, model_id, el_suffix)
        if not pretrain_model.is_file():
            raise FileNotFoundError(
                f'Missing EL checkpoint for downstream IM: {pretrain_model}\n'
                f'Adjust --el-stem / --el-suffix to match weights/EL.'
            )
        logger.info(f'Loading EL init from {pretrain_model}')
        imm_path = imm_model_path.with_stem(f'{imm_model_path.stem}-{model_id}')
        trainer = Trainer(ImmuScope, model_path=imm_path, device=device, logger=logger, **model_cnf['model'])

        train_immuscope_im(trainer, pretrain_model, mhc_name_seq, data_cnf, model_cnf, logger=logger)

        pred_instances, _, _ = trainer.predict(test_loader, model_prefix="")
        labels = test_loader.dataset.labels
        auc_group = calculate_group_auc(labels, pred_instances, test_loader.dataset.mhc_names, min_pos_num=1)
        auc_all = calculate_auc(labels, pred_instances)
        logger.info(f'|**TEST: AUC_GROUP: {auc_group:.4f}**|')
        logger.info(f'|**TEST: AUC_ALL: {auc_all:.4f}**|')

        save_path = Path.joinpath(result_path, f'results_{out_name}_{model_id}.csv')
        peptide_test_data = restore_peptide_sequences(test_loader.dataset.peptide_embedding.reshape(-1, 27))
        res_df = pd.DataFrame({'mhc': test_loader.dataset.mhc_names, 'peptide': peptide_test_data,
                               'label': labels, 'pred': pred_instances})
        res_df.to_csv(save_path, index=False)

        pred_instances_models.append(pred_instances)
        labels_models.append(labels)

    logger.info(f'-----------------Average-----------------')
    save_path = Path.joinpath(result_path, f'results_{out_name}_avg.csv')
    peptide_test_data = restore_peptide_sequences(test_loader.dataset.peptide_embedding.reshape(-1, 27))
    pred_instances_models = np.array(pred_instances_models).mean(axis=0)
    labels_models = np.array(labels_models).mean(axis=0)
    res_df = pd.DataFrame({'mhc': test_loader.dataset.mhc_names, 'peptide': peptide_test_data,
                           'label': labels_models, 'pred': pred_instances_models})
    res_df.to_csv(save_path, index=False)

    auc_group = calculate_group_auc(labels_models, pred_instances_models, test_loader.dataset.mhc_names, min_pos_num=1)
    auc_all = calculate_auc(labels_models, pred_instances_models)
    logger.info('|**========== TEST: AUC_GROUP: {:.4f} =========**|'.format(auc_group))
    logger.info('|**========== TEST: AUC_ALL: {:.4f} =========**|'.format(auc_all))


if __name__ == '__main__':
    main()

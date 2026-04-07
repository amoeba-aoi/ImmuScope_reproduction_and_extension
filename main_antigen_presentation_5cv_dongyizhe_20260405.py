# -*- coding: utf-8 -*-
# 2026/4/5
# dongyizhe
"""
@Time ： 2024/2/29
@Auth ： shenlongchen
Extensions: eval-only, cv-start/end, skip-pretrain-cvs, yaml metric_fn, checkpoint-aware eval.
Uses train_immuscope_el(..., skip_pretrain=True) on selected folds; implemented in ImmuScope/models/trainer_el.py
(kwargs.pop — official main_antigen_presentation_5cv.py does not pass it). For upstream baseline use main_antigen_presentation_5cv.py.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import click
from torch.utils.data.dataloader import DataLoader
from ImmuScope.datasets.datasets import MABags, SinInstanceBag
from ImmuScope.models.trainer_el import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *


def get_data_loader(data_cnf, mhc_name_seq, model_cnf, cv_):
    path_sa = data_cnf['5cv_sa'].replace('_.h5', f'_{cv_}_train.h5')
    path_ma = data_cnf['5cv_ma'].replace('_.h5', f'_{cv_}_train.h5')
    test_path = data_cnf['5cv_sa'].replace('_.h5', f'_{cv_}_test.h5')
    # split train data into train and valid
    train_idx, valid_idx = create_splits(path_sa, split_ratio=0.1, seed=model_cnf['seed'])
    train_loader_sa = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=train_idx),
        batch_size=model_cnf['train']['batch_size'] * 10, shuffle=True, drop_last=True)
    train_loader_ma = DataLoader(
        MABags(path_ma, mhc_name_seq, model_cnf['model']['bag_size']),
        batch_size=model_cnf['train']['batch_size'], shuffle=True, drop_last=True)
    train_loader = [train_loader_sa, train_loader_ma]

    valid_loader = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=valid_idx), batch_size=model_cnf['valid']['batch_size'])

    test_loader = DataLoader(
        SinInstanceBag(test_path, mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])
    # select only positive instances
    test_loader_ma_pos = DataLoader(
        MABags(path_ma, mhc_name_seq, model_cnf['model']['bag_size'], onlyPositive=True),
        batch_size=model_cnf['test']['batch_size'])
    return train_loader, valid_loader, test_loader, test_loader_ma_pos


def test_immuscope_el(trainer, model_cnf, test_path, mhc_name_seq):
    test_loader = DataLoader(SinInstanceBag(test_path, mhc_name_seq, indices=None),
                             batch_size=model_cnf['test']['batch_size'])
    pred_instances, pred_bags, _ = trainer.predict(test_loader)
    return pred_instances, pred_bags, test_loader.dataset.labels, test_loader.dataset.mhc_names


def test_immuscope_el_with_loader(trainer, test_loader, model_prefix='fine-tune-b'):
    pred_instances, pred_bags, _ = trainer.predict(test_loader, model_prefix=model_prefix)
    return (pred_instances, pred_bags, test_loader.dataset.labels[test_loader.dataset.indices],
            test_loader.dataset.mhc_names[test_loader.dataset.indices])


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope-EL.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
@click.option('--eval-only', is_flag=True, default=False,
              help='Skip training; load saved weights and run each fold test only.')
@click.option('--model-prefix', 'model_prefix_override', default=None, type=str,
              help='Checkpoint stem suffix (default: fine-tune-b).')
@click.option('--cv-start', default=0, type=int,
              help='First CV fold index (0..4). Default 0.')
@click.option('--cv-end', default=5, type=int,
              help='One past last fold to run (exclusive), e.g. 5 runs folds 0–4. Not 5 folds ⇒ summary metrics differ from full 5cv.')
@click.option('--skip-pretrain-cvs', 'skip_pretrain_cvs', default='', type=str,
              help='Comma-separated fold indices (0–4) that skip MIL pretrain and load existing *-CV{k}-pretrain.pt. '
                   'Example: "0" to resume only cv:0; leave empty for full pretrain on every fold.')
def main(data_cnf, model_cnf, start_id, num_models, eval_only, model_prefix_override, cv_start, cv_end,
         skip_pretrain_cvs):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]
    model_path = Path(os.path.join(Path(model_cnf['path']), 'EL', f'{model_name}.pt'))
    res_path = Path(data_cnf['results']) / f'{model_name}'
    res_path.mkdir(parents=True, exist_ok=True)

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    if not (0 <= cv_start < cv_end <= 5):
        raise ValueError(f'Need 0 <= cv_start < cv_end <= 5, got cv_start={cv_start}, cv_end={cv_end}')
    if cv_end - cv_start < 5:
        logger.warning(
            'Running a subset of folds only; concatenated preds/labels and final mean metrics are not full 5cv.')

    skip_pt_cvs = frozenset()
    if skip_pretrain_cvs.strip():
        skip_pt_cvs = frozenset(int(x.strip()) for x in skip_pretrain_cvs.split(',') if x.strip())
        for k in skip_pt_cvs:
            if k < 0 or k > 4:
                raise ValueError(f'skip-pretrain-cvs must be indices in 0..4, got {k}')

    all_models_scores, all_models_labels = [], []

    for model_id in range(start_id, start_id + num_models):
        scores_all, labels_all, mhc_groups_all = [], [], []
        for cv_ in range(cv_start, cv_end):
            if eval_only:
                logger.info(f'------------- Eval only model_id: {model_id} - cv: {cv_} ------------')
            else:
                logger.info(f'------------- Start training model_id: {model_id} - cv: {cv_} ------------')

            path_ = model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}')

            loader = get_data_loader(data_cnf, mhc_name_seq, model_cnf, cv_)

            train_path_ms = Path(os.path.join(data_cnf["dataset_ms"], f"{model_name}_{model_id}_cv{cv_}_train.h5"))
            res_path_5cv = Path(res_path, f'{model_name}-5CV-{model_id}-cv{cv_}')

            train_kw = dict(model_cnf['train'])
            metric_fn = train_kw.pop('metric_fn', 'Triplet')
            if cv_ in skip_pt_cvs:
                train_kw['skip_pretrain'] = True
                logger.info(f'cv {cv_}: skip_pretrain enabled (load existing *-CV{cv_}-pretrain.pt)')
            trainer = Trainer(ImmuScope, model_path=path_, device=device, logger=logger, metric_fn=metric_fn,
                                **model_cnf['model'])

            train_loader, valid_loader, test_loader, test_loader_ma_pos = loader
            if not eval_only:
                trainer.train_immuscope_el(train_loader, valid_loader, test_loader, test_loader_ma_pos, train_path_ms,
                                           res_path=res_path_5cv, **train_kw)

            prefix = model_prefix_override.strip() if model_prefix_override else 'fine-tune-b'
            ckpt_path = path_.with_stem(f'{path_.stem}-{prefix}')
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f'Missing checkpoint for eval: {ckpt_path}. Use --model-prefix to set the stem suffix, '
                    f'or train first without --eval-only.')
            logger.info(f'Final test checkpoint suffix: {prefix} ({ckpt_path})')
            pred_instances, pred_bags, labels, mhc_names = test_immuscope_el_with_loader(
                trainer, test_loader, model_prefix=prefix)
            output_res(mhc_names, labels, pred_instances, res_path_5cv, logger=logger)

            scores_all.extend(pred_instances)
            labels_all.extend(labels)
            mhc_groups_all.extend(mhc_names)
            logger.info(f'------------- Finish training model_id: {model_id} - cv: {cv_} ------------\n')

        all_models_scores.append(np.array(scores_all))
        all_models_labels.append(np.array(labels_all))

        scores_test = np.mean(all_models_scores, axis=0)
        data_truth = np.mean(all_models_labels, axis=0)
        auc0_1_test, aupr_test, ppv_test = calculate_all_metrics(data_truth, scores_test)
        logger.info("All Mean Test: AUC0_1: {:.4f} - AUPR: {:.4f} - PPV: {:.4f}".format(
            auc0_1_test, aupr_test, ppv_test))
        logger.info(f'Finish test ---------------------------------- {model_id}')
        res_path_final = Path(res_path, f'{model_name}-5CV-{model_id}')
        output_res(mhc_groups_all, data_truth, scores_test, res_path_final, logger=logger)


if __name__ == '__main__':
    main()

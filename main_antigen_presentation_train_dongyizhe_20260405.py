# -*- coding: utf-8 -*-
# 2026/4/5
# dongyizhe
"""
@Time ： 2024/2/29
@Auth ： shenlongchen
@Description : ImmuScope-EL use all MA (EL) and SA (EL) data to train for antigen presentation prediction
Extensions: eval-only, model-prefix, skip-pretrain, yaml metric_fn; final eval uses pretrain if fine_tune_epochs==0 else fine-tune-b (overridable by --model-prefix). For upstream use main_antigen_presentation_train.py.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import click
from torch.utils.data.dataloader import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # faster conv when input sizes are fixed
from ImmuScope.datasets.datasets import MABags, SinInstanceBag
from ImmuScope.models.trainer_el import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *



def get_data_loader(data_cnf, mhc_name_seq, model_cnf):
    path_sa, path_ma = data_cnf['train_sa'], data_cnf['train_ma']
    train_idx, valid_idx, test_idx = create_splits_train_valid_test(path_sa, train_ratio=0.1, valid_ratio=0.05,
                                                                    test_ratio=0.05, seed=model_cnf['seed'])
    num_workers = model_cnf['train'].get('num_workers', 4)
    pin_memory = torch.cuda.is_available()
    train_loader_sa = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=train_idx),
        batch_size=model_cnf['train']['batch_size'] * 10, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    train_loader_ma = DataLoader(
        MABags(path_ma, mhc_name_seq, model_cnf['model']['bag_size']),
        batch_size=model_cnf['train']['batch_size'], shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    train_loader = [train_loader_sa, train_loader_ma]

    valid_loader = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=valid_idx), batch_size=model_cnf['valid']['batch_size'],
        num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=test_idx), batch_size=model_cnf['test']['batch_size'],
        num_workers=num_workers, pin_memory=pin_memory)
    # select only positive instances
    test_loader_ma_pos = DataLoader(
        MABags(path_ma, mhc_name_seq, model_cnf['model']['bag_size'], onlyPositive=True),
        batch_size=model_cnf['test']['batch_size'], num_workers=num_workers, pin_memory=pin_memory)
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


def el_final_eval_model_prefix(train_cfg):
    """*-fine-tune-b exists only when fine_tune_epochs > 0; else use *-pretrain."""
    ft = train_cfg.get('fine_tune_epochs', 3)
    try:
        ft = int(ft)
    except (TypeError, ValueError):
        ft = 3
    return 'fine-tune-b' if ft > 0 else 'pretrain'


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope-EL.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
@click.option('--eval-only', is_flag=True, default=False,
              help='Skip training; load saved weights and run test split only (same indices as training script).')
@click.option('--model-prefix', 'model_prefix_override', default=None, type=str,
              help='Checkpoint stem suffix; default: fine-tune-b if fine_tune_epochs>0 else pretrain.')
@click.option('--skip-pretrain', is_flag=True, default=False,
              help='Skip MIL pretrain; load existing *-{model_id}-pretrain.pt before SI/FT.')
def main(data_cnf, model_cnf, start_id, num_models, eval_only, model_prefix_override, skip_pretrain):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]
    model_path = Path(os.path.join(Path(model_cnf['path']), 'EL', f'{model_name}.pt'))

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    scores_all, labels_all = [], []
    res_path = Path(data_cnf['results']) / f'{model_name}'
    res_path.mkdir(parents=True, exist_ok=True)

    for model_id in range(start_id, start_id + num_models):
        if eval_only:
            logger.info(f'------------- Eval only model_id: {model_id} ------------')
        else:
            logger.info(f'------------- Start training model_id: {model_id} -  ------------')
        path_ = model_path.with_stem(f'{model_path.stem}-{model_id}')

        loader = get_data_loader(data_cnf, mhc_name_seq, model_cnf)

        train_path_ms = Path(os.path.join(data_cnf["dataset_ms"], f"{model_name}_{model_id}_train.h5"))
        res_path_with_id = Path(res_path, f'{model_name}-{model_id}')

        train_kw = dict(model_cnf['train'])
        metric_fn = train_kw.pop('metric_fn', 'Triplet')
        if skip_pretrain:
            train_kw['skip_pretrain'] = True
        trainer = Trainer(ImmuScope, model_path=path_, device=device, logger=logger, metric_fn=metric_fn,
                            **model_cnf['model'])

        train_loader, valid_loader, test_loader, test_loader_ma_pos = loader
        if not eval_only:
            trainer.train_immuscope_el(train_loader, valid_loader, test_loader, test_loader_ma_pos, train_path_ms,
                                       res_path=res_path_with_id, **train_kw)

        prefix = (
            model_prefix_override.strip()
            if model_prefix_override
            else el_final_eval_model_prefix(model_cnf['train'])
        )
        ckpt_path = path_.with_stem(f'{path_.stem}-{prefix}')
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f'Missing checkpoint for eval: {ckpt_path}. Use --model-prefix to set the stem suffix, '
                f'or train first without --eval-only.')
        logger.info(f'Final test checkpoint suffix: {prefix} ({ckpt_path})')
        pred_i, pred_b, loss = trainer.predict(test_loader, model_prefix=prefix)
        labels = test_loader.dataset.labels[test_loader.dataset.indices]
        mhc_names = test_loader.dataset.mhc_names[test_loader.dataset.indices]
        output_res(mhc_names, labels, pred_i, res_path_with_id, logger=logger)

        logger.info(f'------------- Finish training model_id: {model_id} ------------\n')

        scores_all.append(np.array(pred_i))
        labels_all.append(np.array(labels))

        scores_test = np.mean(scores_all, axis=0)
        data_truth = np.mean(labels_all, axis=0)
        auc0_1_test, aupr_test, ppv_test = calculate_all_metrics(data_truth, scores_test)
        logger.info("All Mean Test: AUC0_1: {:.4f} - AUPR: {:.4f} - PPV: {:.4f}".format(
            auc0_1_test, aupr_test, ppv_test))
        logger.info(f'Finish test ---------------------------------- {model_id}')
        res_path_final = Path.joinpath(res_path, f'results_{model_name}_avg')
        output_res(mhc_names, data_truth, np.mean(scores_all, axis=0), res_path_final, logger=logger)


if __name__ == '__main__':
    main()

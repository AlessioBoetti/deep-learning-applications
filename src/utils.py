import os
from typing import List, Union
import json
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics

from dataloaders import MNIST_Dataset, CIFAR10_Dataset, FashionMNIST_Dataset, NLP_Dataset
from model import Lion


def create_dirs_if_not_exist(dirs: List[str]):
    if not isinstance(dirs, List):
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    else:
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


def setup_folders(args, cfg, run_name):
    # Remember: dir is the folder, path is the... path
    cfg['out_path'] = os.path.join(cfg['results_dir'], run_name)
    create_dirs_if_not_exist([cfg['out_path']])
    if cfg['xai']:
        cfg['xai_path'] = os.path.join(cfg['out_path'], 'xai')
        create_dirs_if_not_exist([cfg['xai_path']])
    return cfg


def load_config(config_filepath):
    with open(config_filepath, 'r') as f:
        cfg = json.load(f)
    return cfg


def save_config(src, dst):
    shutil.copy(src, os.path.join(dst, src))


def setup_optimizer(model, cfg):
    if cfg['optimizer_name'] == 'Lion':
        opt = Lion(model.parameters(), **cfg['optimizer'])
    else:
        opt_class = getattr(optim, cfg['optimizer_name'])  # Select torch.nn.optim class based on optimizer_name
        opt = opt_class(model.parameters(), **cfg['optimizer'])
    return opt


def setup_loss(criterion: str):
    criterion_class = getattr(nn, criterion)
    criterion = criterion_class()
    return criterion


def setup_scheduler(optimizer, cfg, train_loader, epochs):
    scheduler_class = getattr(optim.lr_scheduler, cfg['scheduler_name'])
    scheduler = scheduler_class(optimizer, **cfg['scheduler'], steps_per_epoch=len(train_loader), epochs=epochs)
    return scheduler


def load_dataset(
        data_dir: str,
        dataset_type: str,
        dataset_name: str,
        problem: str = None, 
        n_classes = None,
        normal_class: Union[int, List[int]] = None, 
        img_size: int = None,
        augment: bool = False,
        normalize: bool = False,
        gcn: bool = False,
        gcn_minmax: bool = False,
        filename: str = None,
        train_set_name: str = None,
        test_set_name: str = None,
        val_set_name: str = None,
        model_name: str = None,
        cache_dir: str = None, 
        padding_side: str = None, 
        trunc_side: str = None,
        max_token_len: int = None,
        device: str = None,
        val_size: float = None,
        val_mix: bool = None,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        use_sampler: bool = True,
        multiclass: bool = None, 
    ):

    dataset_type = dataset_type.lower().replace(' ', '')
    dataset_name = dataset_name.lower().replace(' ', '')        
    
    if dataset_type == 'vision':
        dataset_kw = dict(
            root=data_dir,
            problem=problem,
            n_classes=n_classes,
            normal_class=normal_class,
            img_size=img_size,
            augment=augment,
            normalize=normalize,
            gcn=gcn, 
            gcn_minmax=gcn_minmax,
            val_size=val_size,
            val_mix=val_mix,
            val_shuffle=val_shuffle,
            val_shuffle_seed=val_shuffle_seed,
            use_sampler=use_sampler,
            multiclass=multiclass, 
        )
        if dataset_name == 'mnist':
            dataset = MNIST_Dataset(**dataset_kw)
        if dataset_name == 'cifar10':
            dataset = CIFAR10_Dataset(**dataset_kw)
        if dataset_name == 'fashionmnist':
            dataset = FashionMNIST_Dataset(**dataset_kw)
        # if dataset_name == 'mvtec':
        #     dataset = MVTEC_Dataset(**dataset_kw)
    
    elif dataset_type == 'textclassification':
        dataset_kw = dict(
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            filename=filename,
            data_dir=data_dir,
            train_set_name=train_set_name,
            test_set_name=test_set_name,
            val_set_name=val_set_name,
            val_size=val_size,
            model_name=model_name,
            cache_dir=cache_dir, 
            padding_side=padding_side, 
            trunc_side=trunc_side,
            max_token_len=max_token_len,
            device=device,
            val_shuffle=val_shuffle,
            val_shuffle_seed=val_shuffle_seed,
            use_sampler=use_sampler,
        )
        dataset = NLP_Dataset(**dataset_kw)
    
    else:
        raise NotImplementedError()

    return dataset


def load_model(model_path, model, optimizer = None):
    chkpt = torch.load(model_path)
    model.load_state_dict(chkpt["state_dict"])
    if optimizer:
        optimizer.load_state_dict(chkpt['optimizer'])
    return model, optimizer, chkpt


def save_model(state: dict, model_path: str, name: str):
    torch.save(state, f"{model_path}/{name}.pth.tar")


def save_results(path, results: dict, set: str, suffix: str = 'results'):
    with open(f'{path}/{set}_{suffix}.json', 'w') as f:
        json.dump(results, f)


def get_metrics(cfg, wb, device):
    # From https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection
    metric_collection = torchmetrics.MetricCollection([
        torchmetrics.classification.MulticlassAccuracy(num_classes=cfg['output_size']).to(device=device),
        torchmetrics.classification.MulticlassPrecision(num_classes=cfg['output_size']).to(device=device),
        torchmetrics.classification.MulticlassRecall(num_classes=cfg['output_size']).to(device=device),
    ])
    
    # From https://docs.wandb.ai/guides/technical-faq/metrics-and-performance#can-i-log-metrics-on-two-different-time-scales-for-example-i-want-to-log-training-accuracy-per-batch-and-validation-accuracy-per-epoch
    # From https://docs.wandb.ai/guides/track/log/log-summary
    wb.define_metric("val_loss", summary="min")
    wb.define_metric("val_accuracy", summary="max")
    wb.define_metric("best_model_epoch")
    
    return metric_collection


def print_logs(logger, cfg, args = None, init: bool = False, pretrain: bool = False, pretest: bool = False, train: bool = False, test: bool = False):
    if init:
        if args.load_config is not None:
            logger.info('Loaded configuration from "%s".' % args.load_config)
        logger.info('Log filepath: %s.' % cfg['log_filepath'])
        if 'data_dir' in cfg.keys():
            logger.info('Data dir: %s.' % cfg['data_dir'])
        if 'dataset' in cfg.keys():
            logger.info('Dataset: %s' % cfg['dataset']['dataset_name'])
        if cfg['problem'] is not None and cfg['problem'].lower() in ['od', 'ood']:
            logger.info('Normal class: %d' % cfg['dataset']['normal_class'])
            logger.info('Multiclass: %s' % cfg['dataset']['multiclass'])
        if 'dataloader' in cfg.keys():
            if 'num_workers' in cfg['dataloader'].keys():
                logger.info('Number of dataloader workers: %d' % cfg['dataloader']['num_workers'])
        logger.info('Network: %s' % cfg['model_type'])
        logger.info('Computation device: %s' % cfg['device'])

    elif pretrain:
        logger.info('Pretraining optimizer: %s' % cfg['pretrainer_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg['pretrain_optimizer']['lr'])
        logger.info('Pretraining epochs: %d' % cfg['pretraining']['n_epochs'])
        if 'lr_milestone' in cfg['pretrain_optimizer']:
            logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg['pretrain_optimizer']['lr_milestone']))
        logger.info('Pretraining batch size: %d' % cfg['dataloader']['pretrainer_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg['pretrain_optimizer']['weight_decay'])
    
    elif pretest:
        pass
    
    elif train:
        logger.info('Training optimizer: %s' % cfg['optimizer_name'])
        logger.info('Training learning rate: %g' % cfg['optimizer']['lr'])
        logger.info('Training epochs: %d' % cfg['training']['n_epochs'])
        if 'lr_milestone' in cfg['optimizer']:
            logger.info('Training learning rate scheduler milestones: %s' % (cfg['optimizer']['lr_milestone']))
        logger.info('Training batch size: %d' % cfg['dataloader']['batch_size'])
        logger.info('Training weight decay: %g' % cfg['optimizer']['weight_decay'])
    
    elif test:
        pass
    
    return


def epoch_logger(logger, epoch, n_epochs, time, loss, metrics):
    logger.info('Epoch {}/{}'.format(epoch + 1, n_epochs))
    logger.info('  Epoch Train Time: {:.3f}'.format(time))
    logger.info('  Epoch Train Loss: {:.8f}'.format(loss))
    for metric, value in metrics.items():
        logger.info('  Epoch Train {}: {:.4f}'.format(metric, value))


def evaluate_logger(logger, total_time, loss_norm, metric_dict, validation):
    log_str = 'Validation' if validation else 'Test'
    logger.info('  {} Time: {:.3f}'.format(log_str, total_time))
    logger.info('  {} Loss: {:.8f}'.format(log_str, loss_norm))
    for metric, value in metric_dict.items():
        logger.info('  {} {}: {:.4f}'.format(log_str, metric, value))
import os
import sys
import argparse
import logging
import json
import yaml
import random
import numpy as np
from dotenv import load_dotenv, find_dotenv
from typing import Union, List

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from model import GPTLanguageModel
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_config', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    args = parser.parse_args()

    return args


def load_config(config_filepath):
    if config_filepath.endswith('.yaml'):
        with open(config_filepath, 'r') as f:
            cfg = yaml.safe_load(f)
    elif config_filepath.endswith('.json'):
        with open(config_filepath, 'r') as f:
            cfg = json.load(f)
    else:
        raise ValueError('The config file is neither YAML nor JSON. Unknown file type.')
    return cfg


def save_config(config_filepath, cfg):
    with open(config_filepath, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


def setup_folders(args, cfg):
    # Remember: dir is the folder, path is the... path
    results_dir = cfg['results_dir']
    run_name = cfg['name']
    cfg['checkpoint_path'] = os.path.join(results_dir, run_name, 'checkpoint')
    cfg['model_path'] = os.path.join(results_dir, run_name, 'model')
    cfg['logs_path'] = os.path.join(results_dir, run_name, 'logs')
    cfg['xai_path'] = os.path.join(results_dir, run_name, 'xai')
    create_dirs_if_not_exist([cfg['data_dir'], cfg['checkpoint_path'], cfg['model_path'], cfg['logs_path'], cfg['xai_path']])
    return cfg


def setup_logging(logging, cfg):
    # logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_filepath = os.path.join(cfg['logs_path'], cfg['log_file'])
    cfg['log_filepath'] = log_filepath

    handler = logging.FileHandler(log_filepath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('Starting run.')
    logger.info('Logger setup correctly.')

    return logger, cfg


def setup_seed(seed, logger):
    # TODO: Improve randomization to make it global and permanent
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True)
        logger.info('Seed set to %d.' % seed)


def print_logs(logger, cfg, args = None, init: bool = False, pretrain: bool = False, pretest: bool = False, train: bool = False, test: bool = False):
    if init:
        logger.info('Loaded configuration from %s.' % args.load_config)
        logger.info('Log filepath is %s.' % cfg['log_filepath'])
        logger.info('Data dir is %s.' % cfg['data_dir'])
        logger.info('Dataset: %s' % cfg['config']['dataset_name'])
        if cfg['problem'].lower() == 'od':
            logger.info('Normal class: %d' % cfg['normal_class'])
            logger.info('Multiclass: %s' % cfg['multiclass'])
        logger.info('Network: %s' % cfg['config']['architecture'])
        logger.info('Weight decay: %.4f' % cfg['config']['weight_decay'])

        logger.info('Computation device: %s' % cfg['device'])
        logger.info('Number of dataloader workers: %d' % cfg['n_workers_dataloader'])
    elif pretrain:
        logger.info('Pretraining optimizer: %s' % cfg['config']['pretrainer_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg['config']['pretrainer_lr'])
        logger.info('Pretraining epochs: %d' % cfg['config']['pretrainer_n_epochs'])
        if 'pretrainer_lr_milestone' in cfg['config']:
            logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg['config']['pretrainer_lr_milestone']))
        logger.info('Pretraining batch size: %d' % cfg['config']['pretrainer_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg['config']['pretrainer_weight_decay'])
    elif pretest:
        pass
    elif train:
        logger.info('Training optimizer: %s' % cfg['config']['optimizer_name'])
        logger.info('Training learning rate: %g' % cfg['config']['lr'])
        logger.info('Training epochs: %d' % cfg['config']['n_epochs'])
        if 'lr_milestone' in cfg['config']:
            logger.info('Training learning rate scheduler milestones: %s' % (cfg['config']['lr_milestone']))
        logger.info('Training batch size: %d' % cfg['config']['batch_size'])
        logger.info('Training weight decay: %g' % cfg['config']['weight_decay'])
    elif test:
        pass
    return


def save_results(results_path, results: dict, set: str):
    with open(f'{results_path}/{set}_results.json', 'w') as f:
        json.dump(results, f)


def load_dataset(
        name: str, 
        data_dir: str,
        val_size: float = None,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        problem: str = None, 
        normal_class: Union[int, List[int]] = None, 
        multiclass: bool = None, 
        img_size: int = None,
        normalize: bool = False,
        gcn: bool = False,
        gcn_minmax: bool = False,
        augment: bool = False,
        use_sampler: bool = True,
    ):

    dataset_kw = dict(
        root=data_dir,
        val_size=val_size,
        val_shuffle=val_shuffle,
        val_shuffle_seed=val_shuffle_seed,
        problem=problem, 
        normal_class=normal_class, 
        multiclass=multiclass, 
        img_size=img_size,
        normalize=normalize,
        gcn=gcn, 
        gcn_minmax=gcn_minmax,
        augment=augment,
        use_sampler=use_sampler
    )

    if name.lower() == 'mnist':
        dataset = MNIST_Dataset(**dataset_kw)
    if name.lower() == 'cifar10':
        dataset = CIFAR10_Dataset(**dataset_kw)
    # if name.lower() == 'mvtec':
    #     dataset = MVTEC_Dataset(**dataset_kw)
    return dataset


def select_loss_fn(criterion: str):
    criterion = criterion.lower().strip().replace(' ', '').replace('-', '')
    if criterion == 'crossentropyloss':
        return nn.CrossEntropyLoss()
    elif criterion == 'bcewithlogitsloss':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError


def main(args, cfg, wb):
    
    cfg = setup_folders(args, cfg)
    logger, cfg = setup_logging(logging, cfg)
    setup_seed(cfg['seed'], logger)

    device = 'cpu' if not torch.cuda.is_available() else cfg['device']
    wb_cfg = cfg['config']

    print_logs(logger, cfg, args, init=True)


    # Loading data...
    data = args.data if args.data else cfg['data']
    logger.info('Loading %s' % data)
    with open(data, 'r', encoding='utf-8') as f:
        text = f.read()
    logger.info('Data loaded.')

    # Creating Encoder and Decoder for embeddings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }  # string to integer
    itos = { i:ch for i,ch in enumerate(chars) }  # integer to string
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Initializing model...
    logger.info('Initializing %s model.' % wb_cfg['architecture'])
    model = GPTLanguageModel(
        
    ).to(device)
    logger.info('Model initialized.')
    logger.info(model)
    

    # Initializing optimizer...
    logger.info('Initializing %s optimizer.' % wb_cfg['optimizer_name'])
    opt_name = wb_cfg['optimizer_name'].lower()
    lr = wb_cfg['lr']
    wd = wb_cfg['weight_decay']
    if opt_name in ['adam', 'amsgrad']:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, amsgrad=opt_name == 'amsgrad')
    elif opt_name == 'lion':
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler()
    logger.info('Optimizer initialized.')

    criterion = select_loss_fn(wb_cfg['criterion'])
    metric_collection = get_metrics(wb, device)


    # Loading model and optimizer...
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    model_path = None
    if args.load_model:
        cfg['model_loaded_from'] = args.load_model
        model_path = args.load_model
    elif cfg['resume']:
        model_path = cfg['checkpoint_path']
    elif os.listdir(cfg['model_path']):
        model_path = cfg['model_path']
    
    if model_path is not None:
        logger.info('Loading model from %s.' % model_path)
        model, optimizer, start, monitored_value, count = load_model(os.path.join(model_path, 'model.pth.tar'), model, optimizer)
        patience = EarlyStopping('max', wb_cfg['patience'], count, monitored_value) if wb_cfg['patience'] else None
        logger.info('Model loaded.')
    else:
        logger.info('Starting model from scratch.')
        start = 0
        patience = EarlyStopping('max', wb_cfg['patience']) if wb_cfg['patience'] else None
    

    # Pretraining model...
    logger.info('Pretraining: %s' % wb_cfg['pretrain'])
    if wb_cfg['pretrain']:
        print_logs(logger, cfg, pretrain=True)

        # Pretraining...
        # deep_SVDD.pretrain(
        #     dataset, 
        #     optimizer_name=cfg.settings['ae_optimizer_name'],
        #     lr=cfg.settings['ae_lr'],
        #     n_epochs=cfg.settings['ae_n_epochs'],
        #     lr_milestones=[cfg.settings['ae_lr_milestone']],
        #     batch_size=cfg.settings['ae_batch_size'],
        #     weight_decay=cfg.settings['ae_weight_decay'],
        #     device=cfg.settings['device'],
        #     n_jobs_dataloader=cfg.settings['n_jobs_dataloader'])
    

    # Training model...
    logger.info('Training: %s' % wb_cfg['train'])
    if wb_cfg['train']:
        train_loader, val_loader, _, _ = dataset.loaders(train=True, val=True, batch_size=wb_cfg['batch_size'], num_workers=cfg['n_workers_dataloader'], seed=cfg['seed'])
        print_logs(logger, cfg, train=True)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=wb_cfg['max_lr'],
            steps_per_epoch=len(train_loader),
            epochs=wb_cfg['n_epochs'] - start
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=wb_cfg['lr_milestones'], gamma=0.1)

        wb.watch(model, criterion=criterion, log="all", log_graph=True)
        if wb_cfg['clip_grads']:
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        logger.info('Starting training...')
        model, train_results = train(
            train_loader, 
            model, 
            start, 
            wb_cfg['n_epochs'], 
            criterion, 
            optimizer, 
            scheduler, 
            scaler, 
            metric_collection, 
            device, 
            logger, 
            wb, 
            cfg,  
            update_scheduler_on_epoch=False,
            update_scheduler_on_batch=True,
            patience=patience,
            val_loader=val_loader)
        logger.info('Finished training.')

        save_results(cfg['logs_path'], train_results, 'train')


    # Testing model...
    logger.info('Testing: %s' % wb_cfg['test'])
    if wb_cfg['test']:
        transformed_test_loader, org_test_loader = dataset.loaders(train=False, batch_size=wb_cfg['batch_size'], num_workers=cfg['n_workers_dataloader'], seed=cfg['seed'])
        logger.info('Starting testing on transformed test set...')
        test_loss_norm, test_metrics, test_time, test_n_batches, idx_label_scores = evaluate(transformed_test_loader, model, criterion, metric_collection, device, logger, validation=False)
        logger.info('Finished testing.')

        test_results = {
            'test_time': test_time,
            'test_loss': test_loss_norm,
            'test_accuracy': test_metrics['MulticlassAccuracy'],
            'test_precision': test_metrics['MulticlassPrecision'],
            'test_recall': test_metrics['MulticlassRecall'],
            'test_scores': idx_label_scores,
        }
        save_results(cfg['logs_path'], test_results, 'transformed_test')

        logger.info('Starting testing on original test set...')
        test_loss_norm, test_metrics, test_time, test_n_batches, idx_label_scores = evaluate(org_test_loader, model, criterion, metric_collection, device, logger, validation=False)
        logger.info('Finished testing.')

        test_results = {
            'test_time': test_time,
            'test_loss': test_loss_norm,
            'test_accuracy': test_metrics['MulticlassAccuracy'],
            'test_precision': test_metrics['MulticlassPrecision'],
            'test_recall': test_metrics['MulticlassRecall'],
            'test_scores': idx_label_scores,
        }

        save_results(cfg['logs_path'], test_results, 'original_test')


    # plot_results()
    save_config(cfg['logs_path'] + '/config.yaml', cfg)

    if cfg['explain_gradients']:
        backprop_grads = VanillaBackprop(model)
        
        transformed_test_loader, org_test_loader = dataset.loaders(train=False, batch_size=wb_cfg['batch_size'], num_workers=cfg['n_workers_dataloader'], seed=cfg['seed'])
        batch = next(iter(transformed_test_loader))
        img, label, idx = batch
        grads = backprop_grads.generate_gradients(img, label)
        
        # Save colored and grayscale gradients
        save_gradient_images(grads, cfg['xai_path'], 'backprop_grads_color')
        grayscale_grads = convert_to_grayscale(grads)
        save_gradient_images(grayscale_grads, cfg['xai_path'], 'backprop_grads_grayscale')
    
    if cfg['explain_cam']:
        logger.info('Explaining model predictions with Class Activation Mappings...')
        cam = ClassActivationMapping(model, target_layer='hook')

        transformed_test_loader, org_test_loader = dataset.loaders(train=False, batch_size=wb_cfg['batch_size'], num_workers=cfg['n_workers_dataloader'], seed=cfg['seed'])
        iter_loader = iter(transformed_test_loader)
        iter_org_loader = iter(org_test_loader)

        for i in np.arange(10):
            batch = next(iter_loader)
            imgs, labels, idx = batch
            img = imgs[i].unsqueeze(0)
            label = labels[i]
            cams = cam.generate_cam(img, label)

            org_batch = next(iter_org_loader)
            org_img = org_batch[0][i]
            save_class_activation_images(org_img, cams, cfg['xai_path'] + f'/gradcam_{i+1}')
        
        logger.info('Finished explaining model.')


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    args = parse_args()
    if args.load_config:
        cfg = load_config(args.load_config)
    else:
        with open('./config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)

    # From https://docs.wandb.ai/ref/python/init
    wb = wandb.init(
        name=cfg['name'],
        project=cfg['project'],
        group=cfg['group'],
        tags=cfg['tags'],
        notes=cfg['notes'],
        job_type=cfg['job_type'],
        resume=cfg['resume'],
        config=cfg['config']
    )
    
    main(args, cfg, wb)
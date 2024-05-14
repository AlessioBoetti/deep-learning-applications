import os
import argparse
import logging
import yaml
import random
from datetime import timedelta
import numpy as np
from dotenv import load_dotenv, find_dotenv
from typing import Union, List

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from dataloaders import MNIST_Dataset, CIFAR10_Dataset
from model import ConvolutionalNeuralNetwork, Lion
from utils import *
from xai import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_config', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    args = parser.parse_args()

    return args


def setup_folders(args, cfg, run_name):
    # Remember: dir is the folder, path is the... path
    cfg['out_path'] = os.path.join(cfg['results_dir'], run_name)
    create_dirs_if_not_exist([cfg['out_path']])
    if cfg['xai']:
        cfg['xai_path'] = os.path.join(cfg['out_path'], 'xai')
        create_dirs_if_not_exist([cfg['xai_path']])
    return cfg


def setup_logging(logging, cfg):
    # logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_filepath = os.path.join(cfg['out_path'], cfg['log_file'])
    cfg['log_filepath'] = log_filepath

    handler = logging.FileHandler(log_filepath)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')  # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('Starting run.')
    logger.info('Logger setup correctly.')

    return logger, cfg


def setup_seed(seed, logger):
    # TODO: Improve randomization to make it global and permanent
    # When using CUDA, the env var in the .env file comes into play!
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For Multi-GPU, exception safe (https://github.com/pytorch/pytorch/issues/108341)
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info('Seed set to %d.' % seed)


def print_logs(logger, cfg, args = None, init: bool = False, pretrain: bool = False, pretest: bool = False, train: bool = False, test: bool = False):
    if init:
        if args.load_config is not None:
            logger.info('Loaded configuration from "%s".' % args.load_config)
        logger.info('Log filepath: %s.' % cfg['log_filepath'])
        logger.info('Data dir: %s.' % cfg['data_dir'])
        logger.info('Dataset: %s' % cfg['dataset']['dataset_name'])
        if cfg['problem'].lower() == 'od':
            logger.info('Normal class: %d' % cfg['dataset']['normal_class'])
            logger.info('Multiclass: %s' % cfg['dataset']['multiclass'])
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


def load_dataset(
        data_dir: str,
        dataset_name: str, 
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

    if dataset_name.lower() == 'mnist':
        dataset = MNIST_Dataset(**dataset_kw)
    if dataset_name.lower() == 'cifar10':
        dataset = CIFAR10_Dataset(**dataset_kw)
    # if dataset_name.lower() == 'mvtec':
    #     dataset = MVTEC_Dataset(**dataset_kw)
    return dataset


def setup_optimizer(model, cfg):
    if cfg['optimizer_name'] == 'Lion':
        opt = Lion(model.parameters(), **cfg['optimizer'])
    else:
        opt_class = getattr(optim, cfg['optimizer_name'])  # Select torch.nn.optim class based on optimizer_name
        opt = opt_class(model.parameters(), **cfg['optimizer'])
    return opt


def select_loss_fn(criterion: str):
    criterion = criterion.lower().strip().replace(' ', '').replace('-', '')
    if criterion == 'crossentropyloss':
        return nn.CrossEntropyLoss()
    elif criterion == 'bcewithlogitsloss':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError


def setup_scheduler(optimizer, cfg, train_loader, epochs):
    scheduler_class = getattr(optim.lr_scheduler, cfg['scheduler_name'])
    scheduler = scheduler_class(optimizer, **cfg['scheduler'], steps_per_epoch=len(train_loader), epochs=epochs)
    return scheduler


def evaluate(loader, model, criterion, metrics, device, logger, validation: bool):
    model.eval()
    running_loss = 0.0
    val_batches = len(loader)
    idx_label_scores = []
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            metrics(outputs, labels)

            if not validation:
                idx_label_scores += list(zip(idx.cpu().data.numpy().tolist(),
                                        labels.cpu().data.numpy().tolist(),
                                        outputs.cpu().data.numpy().tolist()))
    
    total_time = time.time() - start_time
    loss_norm = running_loss / val_batches
    metric_dict = {metric: float(metrics[metric].compute().cpu().data.numpy() * 100) for metric in metrics.keys()}
    metrics.reset()

    log_str = 'Validation' if validation else 'Test'
    logger.info('  {} Time: {:.3f}'.format(log_str, total_time))
    logger.info('  {} Loss: {:.8f}'.format(log_str, loss_norm))
    for metric, value in metric_dict.items():
        logger.info('  {} {}: {:.4f}'.format(log_str, metric, value)) 
    
    return loss_norm, metric_dict, total_time, val_batches, idx_label_scores


def train(
        loader, 
        model, 
        start, 
        n_epochs, 
        criterion, 
        optimizer, 
        scheduler, 
        scaler,  
        metrics, 
        device, 
        logger,
        wb,
        out_path,
        update_scheduler_on_batch: bool, 
        update_scheduler_on_epoch: bool,
        patience = None,
        val_loader = None, 
        eval_interval: int = 1,
        checkpoint_every: int = 5,
        best: dict = None,
        lr_milestones: List[int] = None, 
    ): 

    train_start_time = time.time()
    total_batches_running = 0
    total_batches = len(loader)
    best_results = best
    model = model.to(device)
    model.train()

    for epoch in range(start, n_epochs):
        loss_epoch = 0.0
        epoch_start_time = time.time()

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            with torch.cuda.amp.autocast():
                # https://pytorch.org/docs/stable/amp.html
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if update_scheduler_on_batch:
                # https://discuss.pytorch.org/t/scheduler-step-after-each-epoch-or-after-each-minibatch/111249
                scheduler.step()
                if lr_milestones is not None:
                    if total_batches_running in lr_milestones:
                        logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch += loss.item()
            total_batches_running += 1

            metrics(outputs, labels)

        epoch_time = time.time() - epoch_start_time
        epoch_loss_norm = loss_epoch / total_batches
        epoch_metrics = {metric: float(metrics[metric].compute().cpu().data.numpy() * 100) for metric in metrics.keys()}
        metrics.reset()
        
        if update_scheduler_on_epoch:
            scheduler.step()
            if lr_milestones is not None:
                if epoch in lr_milestones:
                    logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
        
        logger.info('Epoch {}/{}'.format(epoch + 1, n_epochs))
        logger.info('  Epoch Train Time: {:.3f}'.format(epoch_time))
        logger.info('  Epoch Train Loss: {:.8f}'.format(epoch_loss_norm))
        for metric, value in epoch_metrics.items():
            logger.info('  Epoch Train {}: {:.4f}'.format(metric, value))

        wb_log = {
            'train_loss': epoch_loss_norm,
            'train_accuracy': epoch_metrics['MulticlassAccuracy'],
            'train_precision': epoch_metrics['MulticlassPrecision'],
            'train_recall': epoch_metrics['MulticlassRecall']
        }

        checkpoint = {
            'start': epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        if val_loader and (epoch % eval_interval == 0 or epoch == n_epochs - 1):
            val_loss_norm, val_metrics, val_time, val_n_batches, _ = evaluate(val_loader, model, criterion, metrics, device, logger, validation=True)

            wb_log.update({
                'val_loss': val_loss_norm,
                'val_accuracy': val_metrics['MulticlassAccuracy'],
                'val_precision': val_metrics['MulticlassPrecision'],
                'val_recall': val_metrics['MulticlassRecall']
            })

            if patience:
                # Save best model
                if patience(val_metrics['MulticlassAccuracy']):
                    logger.info('  Found best model, saving model.')
                    best_results = {'best_epoch': epoch + 1}
                    best_results.update({f'best_{key}': value for key, value in wb_log.items()})                                        
                    save_model(checkpoint, out_path, 'model')
                
                checkpoint.update({
                    'max_accuracy': getattr(patience, 'baseline'), 
                    'count': getattr(patience, 'count'), 
                    'best': best_results
                })
            
            model.train()

        wb.log(wb_log)            

        # Save checkpoint
        if epoch % checkpoint_every == 0:
            save_model(checkpoint, out_path, 'checkpoint')
        
        # Early stopping
        if patience and getattr(patience, 'count') == 0:
            logger.info('  Early stopping. Ending training.')
            break

    train_time = time.time() - train_start_time
    logger.info('Training time: %.3f' % train_time)

    train_results = {
        'total_batches': total_batches,
        'total_epochs': n_epochs,
        'train_time': train_time,
        'h-m-s_train_time': str(timedelta(seconds=train_time)),
    }
    train_results.update({f'last_epoch_{key}': value for key, value in wb_log.items()})
    train_results['early_stopping'] = True if patience else False
    if patience:
        train_results.update(best_results)

    return model, train_results


def main(args, cfg, wb, run_name):
    
    cfg = setup_folders(args, cfg, run_name)
    logger, cfg = setup_logging(logging, cfg)
    setup_seed(cfg['seed'], logger)

    device = 'cpu' if not torch.cuda.is_available() else cfg['device']
    cfg['device'] = device

    if wandb.run.resumed:
        logger.info('Resuming experiment.')
    print_logs(logger, cfg, args, init=True)


    # Loading dataset...
    data_dir = args.data_dir if args.data_dir else cfg['data_dir']
    logger.info('Loading dataset from "%s".' % cfg['data_dir'])
    dataset = load_dataset(data_dir, val_shuffle_seed=cfg['seed'], **cfg['dataset'])
    dataloader_kw = dict(seed=cfg['seed'], device='cpu', **cfg['dataloader'])  # https://stackoverflow.com/questions/68621210/runtimeerror-expected-a-cuda-device-type-for-generator-but-found-cpu

    logger.info('Dataset loaded.')


    # Initializing model...
    model_cfg, train_cfg = cfg['model'], cfg['training']
    logger.info('Initializing %s model.' % cfg['model_type'])
    model = ConvolutionalNeuralNetwork(
        **model_cfg,
    ).to(device)
    logger.info('Model initialized.')
    logger.info('Showing model structure:')
    logger.info(model)
    

    # Initializing optimizer...
    logger.info('Initializing %s optimizer.' % cfg['optimizer_name'])
    optimizer = setup_optimizer(model, cfg)
    logger.info('Optimizer initialized.')
    scaler = torch.cuda.amp.GradScaler()
    criterion = select_loss_fn(cfg['criterion'])
    metric_collection = get_metrics(cfg['model'], wb, device)


    # Loading model and optimizer...
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    model_path = None
    if args.load_model:
        model_path = args.load_model
    elif wandb.run.resumed:
        model_path = cfg['out_path'] + '/checkpoint.pth.tar'
    elif 'model.pth.tar' in os.listdir(cfg['out_path']):
        model_path = cfg['out_path'] + '/model.pth.tar'
        cfg['train'] = False
    
    if model_path is not None:
        logger.info('Loading model from "%s".' % model_path)
        model, optimizer, checkpoint = load_model(model_path, model, optimizer)
        start = checkpoint['start']
        if wandb.run.resumed and train_cfg['patience']:
            patience = EarlyStopping('max', train_cfg['patience'], checkpoint['count'], checkpoint['max_accuracy'])
            best = checkpoint['best']
        else:
            patience, best = None, None
        logger.info('Model loaded.')
    else:
        logger.info('Starting model from scratch.')
        start, best = 0, None
        patience = EarlyStopping('max', train_cfg['patience']) if train_cfg['patience'] else None
    
    save_config('config.yaml', cfg['out_path'])  

    # Pretraining model...
    logger.info('Pretraining: %s' % cfg['pretrain'])
    if cfg['pretrain']:
        print_logs(logger, cfg, pretrain=True)
        torch.cuda.empty_cache()

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
    

    # Testing pretrained model...
    logger.info('Testing pretrained model: %s' % cfg['pretest'])
    if cfg['pretest']:
        print_logs(logger, cfg, pretrain=True)

        # ae_idx_label_score, ae_auc = deep_SVDD.test_ae( 
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
    logger.info('Training: %s' % cfg['train'])
    if cfg['train']:
        train_loader, val_loader, _, _ = dataset.loaders(train=True, val=True, **dataloader_kw)
        print_logs(logger, cfg, train=True)
        torch.cuda.empty_cache()

        scheduler = setup_scheduler(optimizer, cfg, train_loader, train_cfg['n_epochs'] - start)

        wb.watch(model, criterion=criterion, log="all", log_graph=True)
        
        if cfg['clip_grads']:
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grads'])

        logger.info('Starting training...')
        model, train_results = train(
            train_loader, 
            model, 
            start, 
            train_cfg['n_epochs'], 
            criterion, 
            optimizer, 
            scheduler, 
            scaler, 
            metric_collection, 
            device, 
            logger, 
            wb, 
            cfg['out_path'],  
            update_scheduler_on_epoch=False,
            update_scheduler_on_batch=True,
            patience=patience,
            val_loader=val_loader,
            eval_interval=train_cfg['eval_interval'],
            checkpoint_every=train_cfg['checkpoint_every'],
            best=best)
        logger.info('Finished training.')

        save_results(cfg['out_path'], train_results, 'train')


    # Testing model...
    logger.info('Testing: %s' % cfg['test'])
    if cfg['test']:
        _, _, transformed_test_loader, org_test_loader = dataset.loaders(train=False, **dataloader_kw)

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
        save_results(cfg['out_path'], test_results, 'transformed_test')

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

        save_results(cfg['out_path'], test_results, 'original_test')


    # plot_results()

    if cfg['explain_gradients']:
        backprop_grads = VanillaBackprop(model)
        
        _, _, transformed_test_loader, org_test_loader = dataset.loaders(train=False, **dataloader_kw)
        batch = next(iter(transformed_test_loader))
        img, label, idx = batch
        grads = backprop_grads.generate_gradients(img, label)
        
        # Save colored and grayscale gradients
        save_gradient_images(grads, cfg['xai_path'], 'backprop_grads_color')
        grayscale_grads = convert_to_grayscale(grads)
        save_gradient_images(grayscale_grads, cfg['xai_path'], 'backprop_grads_grayscale')

        logger.info('Finished explaining model.')
    
    if cfg['explain_cam']:
        logger.info('Explaining model predictions with Class Activation Mappings...')
        cam = ClassActivationMapping(model, target_layer='hook')

        _, _, transformed_test_loader, org_test_loader = dataset.loaders(train=False, **dataloader_kw)
        iter_loader = iter(transformed_test_loader)
        iter_org_loader = iter(org_test_loader)
        batch = next(iter_loader)
        org_batch = next(iter_org_loader)
        imgs, labels, idx = batch
        label_list = [label.item() for label in labels]
        len_imgs = len(imgs)

        for i in np.arange(0, len_imgs, 4):
            img = imgs[i].unsqueeze(0)
            label = labels[i]
            cams = cam.generate_cam(img, label)
            org_img = org_batch[0][i]
            save_class_activation_images(org_img, cams, cfg['xai_path'] + f'/gradcam_{i+1}')
        
        logger.info('Finished explaining predictions with Class Activation Mappings..')

    logger.info('Finished run.')
    logger.info('Closing experiment.')
    print('Finished run. Closing experiment.')


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    args = parse_args()
    if args.load_config:
        cfg = load_config(args.load_config)
    else:
        with open('./config.yaml', 'r') as config_file:
            cfg = yaml.safe_load(config_file)

    # From https://docs.wandb.ai/ref/python/init
    wb = wandb.init(
        config=cfg['config'],
        **cfg['wandb']
    )
    
    run_name = cfg['wandb']['name']
    cfg = cfg['config']
    
    main(args, cfg, wb, run_name)
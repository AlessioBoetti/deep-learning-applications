import sys
sys.path.insert(1, '../src')
import os
import argparse
import logging
import time
import random
from datetime import timedelta
from typing import Union, List
import gc

import yaml
import numpy as np
from dotenv import load_dotenv, find_dotenv
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
# import torch.optim as optim

from model import BERT, EarlyStopping
from utils import *
from xai import *
from adversarial import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_config', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    args = parser.parse_args()

    return args


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


def setup_seed(cfg, logger):
    # TODO: Improve randomization to make it global and permanent
    # When using CUDA, the env var in the .env file comes into play!
    seed = cfg['seed']
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For Multi-GPU, exception safe (https://github.com/pytorch/pytorch/issues/108341)
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = cfg['cuda_benchmark']
        torch.backends.cudnn.deterministic = True
        logger.info('Seed set to %d.' % seed)


def evaluate_llm(loader, model, criterion, metrics, device, logger, validation: bool):
    model.eval()
    running_loss = 0.0
    val_batches = len(loader)
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_masks, labels = batch

            input_ids = input_ids.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            metrics(outputs, labels)
    
    total_time = time.time() - start_time
    loss_norm = running_loss / val_batches
    metric_dict = {metric: float(metrics[metric].compute().cpu().data.numpy() * 100) for metric in metrics.keys()}
    metrics.reset()

    evaluate_logger(logger, total_time, loss_norm, metric_dict, validation)
    
    return loss_norm, metric_dict, total_time, val_batches


def train_llm(
        loader, 
        model, 
        start, 
        n_epochs, 
        criterion, 
        optimizer,
        clip_grads, 
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

        for batch in tqdm(loader, desc=f'Epoch {epoch + 1}'):
            # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            with torch.cuda.amp.autocast():  # https://pytorch.org/docs/stable/amp.html
                outputs = model(input_ids, attention_masks)
                loss = criterion(outputs, labels)
            
            if clip_grads:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grads)

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

            # From: https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#torchmetrics.classification.MulticlassAccuracy
            # The first argument of metrics() can be an int tensor of shape (N, ...) or float tensor of shape (N, C, ..).
            # If preds is a floating point we apply torch.argmax along the C dimension to automatically convert probabilities/logits into an int tensor.
            # This means that since the model returns logits (or softmax if the CrossEntropyLoss is NOT used and we are doing multiclass classification)
            # the class with the higher output value is selected automatically by the metrics() class. Otherwise we should implement argmax ourselves.
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
        
        epoch_logger(logger, epoch, n_epochs, epoch_time, epoch_loss_norm, epoch_metrics)

        # Other metrics: https://towardsdatascience.com/efficient-pytorch-supercharging-training-pipeline-19a26265adae
        wb_log = {
            'train_loss': epoch_loss_norm,
            'train_accuracy': epoch_metrics['MulticlassAccuracy'],
            'train_precision': epoch_metrics['MulticlassPrecision'],
            'train_recall': epoch_metrics['MulticlassRecall'],
            'learning_rate': scheduler.get_last_lr(),
            'epoch_train_time': epoch_time,
        }

        checkpoint = {
            'start': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if val_loader and (epoch % eval_interval == 0 or epoch == n_epochs - 1):
            val_loss_norm, val_metrics, val_time, val_n_batches = evaluate_llm(val_loader, model, criterion, metrics, device, logger, validation=True)

            wb_log.update({
                'val_loss': val_loss_norm,
                'val_accuracy': val_metrics['MulticlassAccuracy'],
                'val_precision': val_metrics['MulticlassPrecision'],
                'val_recall': val_metrics['MulticlassRecall'],
                'val_time': val_time,
            })

            if patience:
                # Save best checkpoint
                if patience(val_metrics['MulticlassAccuracy']):
                    logger.info('  Found best checkpoint, saving checkpoint.')
                    best_results = {'best_epoch': epoch + 1}
                    best_results.update({f'best_{key}': value for key, value in wb_log.items()})                                        
                    save_model(checkpoint, out_path, 'best_checkpoint')
                
                checkpoint.update({
                    'max_accuracy': getattr(patience, 'baseline'), 
                    'count': getattr(patience, 'count'), 
                    'best': best_results,
                })
            
            model.train()

        wb.log(wb_log)            

        # Save checkpoint
        if epoch % checkpoint_every == 0 or epoch == n_epochs - 1:
            save_model(checkpoint, out_path, 'checkpoint')
        
        # Early stopping
        if patience and getattr(patience, 'count') == 0:
            logger.info('  Early stopping. Ending training.')
            break
        
        gc.collect()
        torch.cuda.empty_cache()

    train_time = time.time() - train_start_time
    logger.info('Training time: %.3f' % train_time)

    results = {
        'total_batches': total_batches,
        'total_epochs': n_epochs,
        'train_time': train_time,
        'h-m-s_train_time': str(timedelta(seconds=train_time)),
    }
    results.update({f'last_epoch_{key}': value for key, value in wb_log.items()})
    results['early_stopping'] = True if patience else False
    if patience:
        results.update(best_results)
    
    os.rename(f'{out_path}/best_checkpoint.pth.tar', f'{out_path}/model.pth.tar')

    return model, results


def main(args, cfg, wb, run_name):
    
    cfg = setup_folders(args, cfg, run_name)
    logger, cfg = setup_logging(logging, cfg)
    setup_seed(cfg, logger)

    device = 'cpu' if not torch.cuda.is_available() else cfg['device']
    cfg['device'] = device
    model_cfg, train_cfg = cfg['model'], cfg['training']

    if wandb.run.resumed:
        logger.info('Resuming experiment.')
    print_logs(logger, cfg, args, init=True)


    # Loading dataset...
    data_dir = args.data_dir if args.data_dir else cfg['data_dir']
    logger.info('Loading dataset from "%s".' % data_dir)
    dataset = load_dataset(
        data_dir, 
        model_name=model_cfg['model_name'],
        cache_dir=cfg['hf_cache_dir'],
        device=device,
        val_shuffle_seed=cfg['seed'], 
        **cfg['dataset']
    )
    dataloader_kw = dict(seed=cfg['seed'], device='cpu', **cfg['dataloader'])  # https://stackoverflow.com/questions/68621210/runtimeerror-expected-a-cuda-device-type-for-generator-but-found-cpu
    logger.info('Dataset loaded.')


    # Initializing model...
    logger.info('Initializing {} model.'.format(cfg['model_type']))
    if 'model_name' in model_cfg.keys():
        logger.info('Model version: {}'.format(model_cfg['model_name']))
    model = BERT(
        cache_dir=cfg['hf_cache_dir'],
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
    criterion = setup_loss(cfg['criterion'])
    metric_collection = get_metrics(cfg['model'], wb, device)


    # Loading model and optimizer...
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    model_path = None
    if args.load_model:
        model_path = args.load_model
    elif wandb.run.resumed:
        if 'checkpoint.pth.tar' in os.listdir(cfg['out_path']):
            model_path = cfg['out_path'] + '/checkpoint.pth.tar'
        else:
            logger.info('Resuming run, but no checkpoint found.')
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
    if cfg['pretrain']:
        logger.info('Pretraining.')
        print_logs(logger, cfg, pretrain=True)
        torch.cuda.empty_cache()

        # TODO: Implement from DL PW
    

    # Testing pretrained model...
    if cfg['pretest']:
        logger.info('Testing pretrained model.')
        print_logs(logger, cfg, pretrain=True)

        # TODO: Implement from DL PW


    # Training model...
    if cfg['train']:
        logger.info('Training.')
        train_loader, val_loader, _, _ = dataset.loaders(train=True, val=True, **dataloader_kw)
        print_logs(logger, cfg, train=True)
        torch.cuda.empty_cache()

        scheduler = setup_scheduler(optimizer, cfg, train_loader, train_cfg['n_epochs'] - start)

        wb.watch(model, criterion=criterion, log="all", log_graph=True)
        
        logger.info('Starting training...')
        model, train_results = train_llm(
            train_loader, 
            model, 
            start, 
            train_cfg['n_epochs'], 
            criterion, 
            optimizer, 
            train_cfg['clip_grads'],
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
            best=best
        )
        logger.info('Finished training.')

        save_results(cfg['out_path'], train_results, 'train')


    # Testing model...
    if cfg['test']:
        logger.info('Testing.')
        _, _, test_loader, _ = dataset.loaders(train=False, **dataloader_kw)

        logger.info('Starting testing on test set...')
        test_loss_norm, test_metrics, test_time, test_n_batches = evaluate_llm(test_loader, model, criterion, metric_collection, device, logger, validation=False)
        logger.info('Finished testing.')

        test_results = {
            'test_time': test_time,
            'test_loss': test_loss_norm,
            'test_accuracy': test_metrics['MulticlassAccuracy'],
            'test_precision': test_metrics['MulticlassPrecision'],
            'test_recall': test_metrics['MulticlassRecall'],
        }
        save_results(cfg['out_path'], test_results, 'test')


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
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        len_imgs = len(imgs)

        for i in np.arange(0, len_imgs, 4):
            img = imgs[i].unsqueeze(0)
            label = labels[i]
            cams = cam.generate_cam(img, target_class=label, device=device)
            org_img = org_batch[0][i]
            save_class_activation_images(org_img, cams, cfg['xai_path'] + f'/gradcam_{i+1}')


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
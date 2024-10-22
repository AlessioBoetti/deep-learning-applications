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

from model import GPTLanguageModel, EarlyStopping
from utils import *
from xai import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_config', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
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


def train_val_test_split(data, split_size):
    n = int(split_size*len(data))
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data


def get_batch(train_data, val_data, split, block_size, batch_size, device):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, criterion, logger, device):
    model.eval()
    losses = {}
    start_time = time.time()
    for split in ['train', 'val']:
        running_loss = 0.0
        for k in range(eval_iters):
            inputs, targets = get_batch(train_data, val_data, split, block_size, batch_size, device)
            logits = model(inputs, device)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = criterion(logits, targets)
            running_loss += loss.item()
        losses[split] = running_loss / eval_iters
        logger.info('  Evaluation {} Loss: {:.8f}'.format(split, losses[split]))
    total_time = time.time() - start_time
    logger.info('  Evaluation Time: {:.3f}'.format(total_time))
    return losses, total_time


def train(
        model, 
        train_data,
        val_data,
        block_size, 
        batch_size,
        start, 
        n_epochs, 
        criterion, 
        optimizer,
        clip_grads,
        scaler,   
        device, 
        logger,
        wb,
        out_path,
        patience = None,
        eval_interval: int = 1,
        eval_iters: int = 100,
        checkpoint_every: int = 5,
        best: dict = None,
        # lr_milestones: List[int] = None, 
    ): 

    train_start_time = time.time()
    # total_batches_running = 0
    # total_batches = len(loader)
    best_results = best
    model = model.to(device)
    model.train()

    torch.cuda.empty_cache()

    for epoch in range(start, n_epochs):
        loss_epoch = 0.0
        epoch_start_time = time.time()

        inputs, targets = get_batch(train_data, val_data, 'train', block_size, batch_size, device)        
        optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        with torch.cuda.amp.autocast():  # https://pytorch.org/docs/stable/amp.html
            logits = model(inputs, device)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = criterion(logits, targets)
        
        if clip_grads:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grads)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # if update_scheduler_on_batch:
        #         # https://discuss.pytorch.org/t/scheduler-step-after-each-epoch-or-after-each-minibatch/111249
        #         scheduler.step()
        #         if lr_milestones is not None:
        #             if total_batches_running in lr_milestones:
        #                 logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        loss_epoch += loss.item()
        # total_batches_running += 1

        # metrics(outputs, labels)

        epoch_time = time.time() - epoch_start_time
        # epoch_loss_norm = loss_epoch / total_batches
        # epoch_metrics = {metric: float(metrics[metric].compute().cpu().data.numpy() * 100) for metric in metrics.keys()}
        # metrics.reset()
        
        # if update_scheduler_on_epoch:
        #     scheduler.step()
        #     if lr_milestones is not None:
        #         if epoch in lr_milestones:
        #             logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        epoch_logger(logger, epoch, n_epochs, epoch_time, loss_epoch)

        # Other metrics: https://towardsdatascience.com/efficient-pytorch-supercharging-training-pipeline-19a26265adae
        wb_log = {
            'epoch_train_loss': loss_epoch,
            # 'train_accuracy': epoch_metrics['MulticlassAccuracy'],
            # 'train_precision': epoch_metrics['MulticlassPrecision'],
            # 'train_recall': epoch_metrics['MulticlassRecall'],
            'epoch_train_time': epoch_time,
        }

        checkpoint = {
            'start': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if epoch % eval_interval == 0 or epoch == n_epochs - 1:
            losses, val_time = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, criterion, logger, device)

            wb_log.update({
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'val_time': val_time,
            })

            if patience:  # Save best model
                if patience(losses['val']):
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
        # 'total_batches': total_batches,
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
    data = torch.tensor(encode(text), dtype=torch.long)


    # Initializing model...
    logger.info('Initializing {} model.'.format(cfg['model_type']))
    if 'model_name' in model_cfg.keys():
        logger.info('Model version: {}'.format(model_cfg['model_name']))
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        **model_cfg,
    ).to(device)
    logger.info('Model initialized.')
    logger.info('Showing model structure:')
    logger.info(model)
    logger.info(f'Number of parameters: {sum(p.numel() for p in model.parameters())/1e6} M')
    

    # Initializing optimizer...
    logger.info('Initializing %s optimizer.' % cfg['optimizer_name'])
    optimizer = setup_optimizer(model, cfg)
    logger.info('Optimizer initialized.')
    scaler = torch.cuda.amp.GradScaler()
    criterion = setup_loss(cfg['criterion'])


    # Loading model and optimizer...
    model, optimizer, start, patience, best, cfg = load_or_start_model(model, optimizer, args, wandb, cfg, train_cfg, device, logger)


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
        train_data, val_data = train_val_test_split(data, split_size=0.9)
        print_logs(logger, cfg, train=True)

        wb.watch(model, criterion=criterion, log="all", log_graph=True)

        logger.info('Starting training...')
        model, train_results = train( 
            model, 
            train_data,
            val_data,
            model_cfg['block_size'], 
            cfg['dataloader']['batch_size'],
            start, 
            train_cfg['n_epochs'], 
            criterion, 
            optimizer,
            train_cfg['clip_grads'],
            scaler, 
            device, 
            logger, 
            wb, 
            cfg['out_path'],  
            patience=patience,
            eval_interval=train_cfg['eval_interval'],
            checkpoint_every=train_cfg['checkpoint_every'],
            best=best
        )
        logger.info('Finished training.')

        save_results(cfg['out_path'], train_results, 'train')


    # Generating text...
    logger.info('Generation: %s' % cfg['generate'])
    if cfg['generate']:
        logger.info('Started generation.')
        start_time = time.time()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        # print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
        with open(f"{cfg['out_path']}/output.txt", 'w') as f:
            f.write(
                decode(
                    model.generate(
                        context, 
                        max_new_tokens=cfg['generation']['max_new_tokens'], 
                        block_size=model_cfg['block_size'], 
                        device=device
                    )[0].tolist()
                )
            )
        logger.info(f'Generation time: {time.time() - start_time}')
        logger.info('Finished generating.')


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
import sys
sys.path.insert(1, '../src')
import os
import argparse
import logging
import time
import random
from typing import Union, List
import gc

import yaml
import numpy as np
from dotenv import load_dotenv, find_dotenv
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn

from model import ConvolutionalNeuralNetwork
from utils import *
from xai import *
from ood import *
import ood
from adversarial import *
from plot_utils import *


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
        torch.use_deterministic_algorithms(cfg['deterministic'], warn_only=True)
        torch.backends.cudnn.benchmark = cfg['cuda_benchmark']
        torch.backends.cudnn.deterministic = True
        logger.info('Seed set to %d.' % seed)


def evaluate_adv_add(
        loader, 
        model, 
        criterion, 
        metrics, 
        device, 
        logger, 
        scaler, 
        adv_cfg,
        metrics_adv,
        validation: bool = True,
    ):
    # Not sure if this function is needed

    model.eval()
    running_loss = 0.0
    val_batches = len(loader)
    idx_label_scores = []
    idx_label_scores_adv = []
    start_time = time.time()

    for batch in loader:
        inputs, labels, idx = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        delta = attack(model, inputs, labels, scaler=scaler, **adv_cfg)
        outputs_adv = model(inputs + delta)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss_adv = criterion(outputs_adv, labels)
        loss = loss + loss_adv

        running_loss += loss.item()
        metrics(outputs, labels)
        metrics_adv(outputs_adv, labels)

        if not validation:
            idx_label_scores += list(zip(idx.cpu().data.numpy().tolist(),
                                    labels.cpu().data.numpy().tolist(),
                                    outputs.cpu().data.numpy().tolist()))
            idx_label_scores_adv += list(zip(idx.cpu().data.numpy().tolist(),
                                    labels.cpu().data.numpy().tolist(),
                                    outputs_adv.cpu().data.numpy().tolist()))
    
    total_time = time.time() - start_time
    loss_norm = running_loss / val_batches
    metric_dict = {metric: float(metrics[metric].compute().cpu().data.numpy() * 100) for metric in metrics.keys()}
    metrics.reset()
    metric_adv_dict = {metric: float(metrics_adv[metric].compute().cpu().data.numpy() * 100) for metric in metrics_adv.keys()}
    metrics_adv.reset()

    evaluate_logger(logger, total_time, loss_norm, metric_dict, validation, metric_adv_dict)
    
    return loss_norm, metric_dict, total_time, val_batches, idx_label_scores, metric_adv_dict, idx_label_scores_adv


def evaluate(
        loader, 
        model, 
        criterion, 
        metrics, 
        device, 
        logger, 
        validation: bool = True,
        scaler=None,
        adv_cfg=None, 
    ):

    model.eval()
    running_loss = 0.0
    val_batches = len(loader)
    idx_label_scores = []
    start_time = time.time()

    
    for batch in loader:
        inputs, labels, idx = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if adv_cfg:
            delta = attack(model, inputs, labels, scaler=scaler, **adv_cfg)
            inputs = inputs + delta
        with torch.no_grad():
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

    evaluate_logger(logger, total_time, loss_norm, metric_dict, validation, True if adv_cfg else False)
    
    return loss_norm, metric_dict, total_time, val_batches, idx_label_scores


def train(
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
        val_loader=None, 
        eval_interval: int = 1,
        checkpoint_every: int = 5,
        patience=None,
        best: dict = None,
        lr_milestones: List[int] = None,
        adv_cfg=None,
        adv_add: bool = True,
        metrics_adv=None,
    ): 

    train_start_time = time.time()
    total_batches_running = 0
    total_batches = len(loader)
    best_results = best
    model = model.to(device)
    model.train()

    torch.cuda.empty_cache()
    
    for epoch in range(start, n_epochs):
        loss_epoch = 0.0
        epoch_start_time = time.time()

        for batch in tqdm(loader, desc=f'Epoch {epoch + 1}'):  # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            with torch.cuda.amp.autocast():  # https://pytorch.org/docs/stable/amp.html
                if adv_cfg:
                    model.eval()
                    delta = attack(model, inputs, labels, scaler=scaler, **adv_cfg)
                    model.train()
                    if adv_add:
                        outputs = model(inputs)
                        outputs_adv = model(inputs + delta)
                    else:
                        outputs = model(inputs + delta)
                    optimizer.zero_grad(set_to_none=True)  # to reset gradients computed during attack
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                if adv_add:
                    loss_adv = criterion(outputs_adv, labels)
                    loss = loss + loss_adv
            
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
            if adv_add:
                metrics_adv(outputs_adv, labels) 
            
        epoch_time = time.time() - epoch_start_time
        epoch_loss_norm = loss_epoch / total_batches
        epoch_metrics = {metric: float(metrics[metric].compute().cpu().data.numpy() * 100) for metric in metrics.keys()}
        metrics.reset()

        if adv_add:
            epoch_metrics_adv = {metric: float(metrics_adv[metric].compute().cpu().data.numpy() * 100) for metric in metrics_adv.keys()}
            metrics_adv.reset()
        
        if update_scheduler_on_epoch:
            scheduler.step()
            if lr_milestones is not None:
                if epoch in lr_milestones:
                    logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
        
        epoch_logger(logger, epoch, n_epochs, epoch_time, epoch_loss_norm, epoch_metrics, epoch_metrics_adv if adv_add else None)

        # Other metrics: https://towardsdatascience.com/efficient-pytorch-supercharging-training-pipeline-19a26265adae
        wb_log = {
            'train_loss': epoch_loss_norm,
            'train_accuracy': epoch_metrics['MulticlassAccuracy'],
            'train_precision': epoch_metrics['MulticlassPrecision'],
            'train_recall': epoch_metrics['MulticlassRecall'],
            'learning_rate': scheduler.get_last_lr(),
            'epoch_train_time': epoch_time,
        }
        if adv_add:
            wb_log.update({
                'adv_train_accuracy': epoch_metrics_adv['MulticlassAccuracy'],
                'adv_train_precision': epoch_metrics_adv['MulticlassPrecision'],
                'adv_train_recall': epoch_metrics_adv['MulticlassRecall'],
            })

        checkpoint = {
            'start': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if val_loader and (epoch % eval_interval == 0 or epoch == n_epochs - 1):
            eval_kw = dict(loader=val_loader, model=model, criterion=criterion, metrics=metrics, device=device, logger=logger)
            val_loss_norm, val_metrics, val_time, _, _ = evaluate(**eval_kw)
            
            wb_log.update({
                'val_loss': val_loss_norm,
                'val_accuracy': val_metrics['MulticlassAccuracy'],
                'val_precision': val_metrics['MulticlassPrecision'],
                'val_recall': val_metrics['MulticlassRecall'],
                'val_time': val_time,
            })

            if adv_cfg:
                adv_val_loss_norm, adv_val_metrics, adv_val_time, _, _ = evaluate(scaler=scaler, adv_cfg=adv_cfg, **eval_kw)

                wb_log.update({
                    'adv_val_loss': adv_val_loss_norm,
                    'adv_val_accuracy': adv_val_metrics['MulticlassAccuracy'],
                    'adv_val_precision': adv_val_metrics['MulticlassPrecision'],
                    'adv_val_recall': adv_val_metrics['MulticlassRecall'],
                    'adv_val_time': adv_val_time,
                    'tot_val_loss': val_loss_norm + adv_val_loss_norm,
                })

            if patience:  # Save best model
                patience_metric = adv_val_metrics['MulticlassAccuracy'] if adv_cfg else val_metrics['MulticlassAccuracy']
                if patience(patience_metric):
                    logger.info('  Found best model, saving model.')
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

    results = format_train_results(total_batches, n_epochs, train_time, wb_log, patience, best_results if patience else None)

    os.rename(f'{out_path}/best_checkpoint.pth.tar', f'{out_path}/model.pth.tar')

    return model, results


def test(loader, model, criterion, metrics, device, logger, dataset, name, scaler=None, adv_cfg=None, alpha: int = None):
    loss, metrics, total_time, _, idx_label_scores = evaluate(loader, model, criterion, metrics, device, logger, validation=False, scaler=scaler, adv_cfg=adv_cfg)
    log_str = f'Finished testing with alpha set to {alpha}' if alpha else 'Finished testing.'
    logger.info(log_str)

    results = {
        'time': total_time,
        'loss': loss,
        'accuracy': metrics['MulticlassAccuracy'],
        'precision': metrics['MulticlassPrecision'],
        'recall': metrics['MulticlassRecall'],
        'scores': idx_label_scores,
    }
    # save_results(cfg['out_path'], results, name)
    plot_results(idx_label_scores, cfg['out_path'], name, dataset.train_set.classes, metrics=results, eps=alpha)
    return idx_label_scores


def main(args, cfg, wb, run_name):
    
    cfg = setup_folders(args, cfg, run_name)
    logger, cfg = setup_logging(logging, cfg)
    setup_seed(cfg, logger)

    device = 'cpu' if not torch.cuda.is_available() else cfg['device']
    cfg['device'] = device
    model_cfg, train_cfg, adv_cfg = cfg['model'], cfg['training'], cfg['adversarial']
    adv_add = cfg['adversarial_add'] if adv_cfg else False

    if wandb.run.resumed:
        logger.info('Resuming experiment.')
    print_logs(logger, cfg, args, init=True)


    # Loading dataset...
    data_dir = args.data_dir if args.data_dir else cfg['data_dir']
    logger.info('Loading dataset from "%s".' % data_dir)
    dataset = load_dataset(
        data_dir, 
        device=device,
        val_shuffle_seed=cfg['seed'], 
        **cfg['dataset']
    )
    if cfg['problem'] is not None:
        if cfg['problem'] == 'OOD':
            ood_dataset = load_dataset(
                data_dir,
                device=device,
                val_shuffle_seed=cfg['seed'],
                problem=cfg['problem'],
                **cfg['ood_dataset']
            )
        else:
            raise NotImplementedError()
    logger.info('Dataset loaded.')

    dataloader_kw = dict(seed=cfg['seed'], device='cpu', **cfg['dataloader'])  # https://stackoverflow.com/questions/68621210/runtimeerror-expected-a-cuda-device-type-for-generator-but-found-cpu
    train_loader, val_loader, test_loader, org_test_loader = dataset.loaders(train=True, val=True, **dataloader_kw)


    # Initializing model...
    logger.info('Initializing {} model.'.format(cfg['model_type']))
    if 'model_name' in model_cfg.keys():
        logger.info('Model version: {}'.format(model_cfg['model_name']))
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
    criterion = setup_loss(cfg['criterion'])
    metric_collection = get_metrics(cfg['model'], wb, device)
    if adv_add:
        metric_collection_adv = get_metrics(cfg['model'], wb, device)

    if adv_cfg:
        adv_cfg.update(dict(
            normalize=cfg['dataset']['normalize'],
            dataset_name=cfg['dataset']['dataset_name'],
            criterion=criterion,  
            device=device, 
            logger=logger,
        ))

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
        print_logs(logger, cfg, train=True)

        scheduler = setup_scheduler(optimizer, cfg, train_loader, train_cfg['n_epochs'] - start)

        wb.watch(model, criterion=criterion, log='all', log_graph=True)      
        
        logger.info('Starting training...')
        model, train_results = train(
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
            val_loader=val_loader,
            eval_interval=train_cfg['eval_interval'],
            checkpoint_every=train_cfg['checkpoint_every'],
            patience=patience,
            best=best,
            adv_cfg=adv_cfg,
            adv_add=adv_add,
            metrics_adv=metric_collection_adv if adv_add else None,
        )
        logger.info('Finished training.')

        save_results(cfg['out_path'], train_results, 'train')


    if cfg['show_adv_imgs']:
        logger.info('Plotting original and adversarial images...')

        norm_stats = dataset.norm_stats
        inv = NormalizeInverse(norm_stats[0], norm_stats[1])
        classes = dataset.test_set.classes
        adv_imgs_path = f"{cfg['out_path']}/imgs_adv_samples"
        create_dirs_if_not_exist(adv_imgs_path)

        M, N = 5, 9

        # From https://github.com/bethgelab/foolbox/issues/74
        # If model is in train mode, dropout and batch norm will constantly change the network and make finding an adversary quite difficult
        # model.train()
        model.eval()

        for batch in test_loader:
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            break
        
        before = inputs.clone().detach()

        with torch.no_grad():
            outputs = model(inputs)
        imgs = inv(inputs)
        plot_images(imgs, labels, outputs, M, N, adv_imgs_path, classes)

        for alpha in np.arange(*cfg['show_alpha_range']):
            adv_cfg.update({'alpha': alpha, 'fast': False})
            delta = attack(model, inputs, labels, scaler=scaler, **adv_cfg)
            with torch.no_grad():
                outputs = model(inputs + delta)
            adv_imgs = inv(inputs + delta)
            plot_images(adv_imgs, labels, outputs, M, N, adv_imgs_path, classes, adv=True, alpha=alpha, n=f'_{alpha}')

            after = (inputs + delta).clone().detach()
            diffs = torch.abs(after - before)
            diffs_inv = inv(diffs)
            
            plot_images(diffs_inv, labels, outputs, M, N, adv_imgs_path, classes, diff=True, alpha=alpha, n=f'_{alpha}')

            diffs_flat = diffs.flatten()
            plt.figure()
            plt.hist(diffs_flat.cpu().numpy())
            plt.title('Diffs')
            plt.savefig(f'{adv_imgs_path}/diffs_hist_{alpha}.png')
            plt.close()
        
        logger.info('Finished plotting images.')


    # Testing model...
    if cfg['test']:
        logger.info('Testing.')
        logger.info('Starting testing on test set...')
        idx_label_scores = test(test_loader, model, criterion, metric_collection, device, logger, dataset, 'test')


    # Testing model on original dataset...
    if cfg['test_original']:
        logger.info('Testing on original dataset.')
        logger.info('Starting testing on original (not transformed) test set...')
        org_idx_label_scores = test(org_test_loader, model, criterion, metric_collection, device, logger, dataset, 'test_original')


    # Testing model on adversarial examples...
    if cfg['test_adversarial']:
        logger.info('Testing on adversarial examples.')
        logger.info('Starting testing on adversarial test set with multiple alpha values...')

        for alpha in np.arange(*cfg['alpha_range']):
            logger.info(f'Alpha set to {alpha}.')
            adv_cfg.update({'alpha': alpha, 'fast': False})
            adv_idx_label_scores = test(org_test_loader, model, criterion, metric_collection, device, logger, dataset, 'test_adversarial', scaler, adv_cfg, alpha)
            if cfg['test']:
                plot_results(idx_label_scores, cfg['out_path'], 'adv', ood_idx_label_scores=adv_idx_label_scores, eps=alpha)


    # Other testing...
    if cfg['test_ood']:
        logger.info('Testing on OOD dataset.')
        _, _, ood_test_loader, _ = ood_dataset.loaders(train=False, **dataloader_kw)

        logger.info('Starting testing on OOD test set...')
        ood_idx_label_scores = test(ood_test_loader, model, criterion, metric_collection, device, logger, dataset, 'test_ood')
        if cfg['test']:
            plot_results(idx_label_scores, cfg['out_path'], 'ood', ood_idx_label_scores=ood_idx_label_scores)

        postprocess_kw = dict(criterion=criterion, scaler=scaler, device=device, adv_cfg=adv_cfg)

        if cfg['postprocess']:                
            for method_name in cfg['postprocess']:
                logger.info(f'Applying {method_name} postprocessing method for OOD detection...')
                method_class = getattr(ood, method_name)
                method_name = f"{method_name.lower().replace('postprocessor', '')}"

                if 'odin' in method_name and cfg['odin_gridsearch']:
                    for T in np.arange(*cfg['odin_temperatures']):
                        for alpha in np.arange(*cfg['odin_alphas']):
                            method = method_class(T, alpha)
                            id_label_scores, ood_label_scores = get_ood_scores(model, test_loader, ood_test_loader, method.postprocess, **postprocess_kw)
                            plot_results(id_label_scores, cfg['out_path'], 'ood', ood_idx_label_scores=ood_label_scores, postprocess=f'{method_name}-{T}-{alpha}')
                else:
                    method = method_class()
                    id_label_scores, ood_label_scores = get_ood_scores(model, test_loader, ood_test_loader, method.postprocess, **postprocess_kw)
                    plot_results(id_label_scores, cfg['out_path'], 'ood', ood_idx_label_scores=ood_label_scores, postprocess=method_name)
                logger.info(f'Finished postprocessing with {method_name} method.')
            
        if cfg['cea']:                
            for method_name in cfg['cea_postprocess']:
                logger.info(f'Applying {method_name} postprocessing method with CEA for OOD detection...')
                method_class = getattr(ood, method_name)
                method_name = f"{method_name.lower().replace('postprocessor', '')}" + '_cea'

                if 'odin' in method_name and cfg['cea_odin_gridsearch']:
                    for T in np.arange(*cfg['cea_odin_temperatures']):
                        for alpha in np.arange(*cfg['cea_odin_alphas']):
                            method = method_class(T, alpha)
                            cea = CEA(model, method, val_loader, criterion, scaler, adv_cfg, device, cfg['cea']['percentile_top'], cfg['cea']['addition_coef'])
                            id_label_scores, ood_label_scores = get_ood_scores(model, test_loader, ood_test_loader, cea.postprocess, **postprocess_kw)
                            plot_results(id_label_scores, cfg['out_path'], 'ood', ood_idx_label_scores=ood_label_scores, postprocess=f'{method_name}-{T}-{alpha}')
                else:
                    method = method_class()
                    cea = CEA(model, method, val_loader, criterion, scaler, adv_cfg, device, cfg['cea']['percentile_top'], cfg['cea']['addition_coef'])
                    id_label_scores, ood_label_scores = get_ood_scores(model, test_loader, ood_test_loader, cea.postprocess, **postprocess_kw)
                    plot_results(id_label_scores, cfg['out_path'], 'ood', ood_idx_label_scores=ood_label_scores, postprocess=method_name)
                logger.info(f'Finished postprocessing with {method_name} method with CEA.')


    if cfg['explain_gradients']:
        logger.info('Explaining model predictions with vanilla gradients...')
        explain_vanilla_gradients(model, test_loader, cfg['xai_path'])
        logger.info('Finished explaining model.')
    

    if cfg['explain_cam']:
        logger.info('Explaining model predictions with Class Activation Mappings...')
        explain_cams(model, test_loader, org_test_loader, cfg['xai_path'], device)
        logger.info('Finished explaining predictions with Class Activation Mappings..')


    logger.info('Finished run.')
    logger.info('Closing experiment.')
    print('Finished run. Closing experiment.')


if __name__ == '__main__':
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
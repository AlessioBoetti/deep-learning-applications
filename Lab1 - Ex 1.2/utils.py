import os
import time
from typing import List
import json
import shutil
import matplotlib.pyplot as plt

import torch
import torchmetrics


def create_dirs_if_not_exist(dirs: List[str]):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def load_config(config_filepath):
    with open(config_filepath, 'r') as f:
        cfg = json.load(f)
    return cfg


def save_config(src, dst):
    shutil.copy(src, os.path.join(dst, src))


def load_model(model_path, model, optimizer = None):
    chkpt = torch.load(model_path)
    model.load_state_dict(chkpt["state_dict"])
    if optimizer:
        optimizer.load_state_dict(chkpt['optimizer'])
    return model, optimizer, chkpt


def save_model(state: dict, model_path: str, name: str):
    torch.save(state, f"{model_path}/{name}.pth.tar")


def save_results(path, results: dict, set: str):
    with open(f'{path}/{set}_results.json', 'w') as f:
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


def save_plot(train_l, train_a, test_l, test_a):
    plt.plot(train_a, '-')
    plt.plot(test_a, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid accuracy')
    plt.savefig('result/accuracy')
    plt.close()

    plt.plot(train_l, '-')
    plt.plot(test_l, '-')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig('result/losses')
    plt.close()


class EarlyStopping:
    def __init__(self, mod, patience, count=None, baseline=None):
        self.patience = patience
        self.count = patience if count is None else count
        if mod == 'max':
            self.baseline = 0
            self.operation = self.max
        if mod == 'min':
            self.baseline = baseline
            self.operation = self.min

    def max(self, monitored_value):
        if monitored_value > self.baseline:
            self.baseline = monitored_value
            self.count = self.patience
            return True
        else:
            self.count -= 1
            return False

    def min(self, monitored_value):
        if monitored_value < self.baseline:
            self.baseline = monitored_value
            self.count = self.patience
            return True
        else:
            self.count -= 1
            return False

    def __call__(self, monitored_value):
        return self.operation(monitored_value)
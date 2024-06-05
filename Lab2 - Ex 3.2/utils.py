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
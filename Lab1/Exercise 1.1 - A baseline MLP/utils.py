import os
from typing import List
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torchmetrics
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import MNIST


def create_dirs_if_not_exist(dirs: List[str]):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def save_checkpoint(state, model_path):
    torch.save(state, f"{model_path}/model.pth.tar")


def load_model(model_path, model, optimizer = None):
    chkpt = torch.load(model_path)
    model.load_state_dict(chkpt["state_dict"])
    if optimizer:
        optimizer.load_state_dict(chkpt['optimizer'])
    return model, optimizer, chkpt['start'], chkpt['max_accuracy'], chkpt['count']


def get_metrics(wb, device):
    # From https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection
    metric_collection = torchmetrics.MetricCollection([
        torchmetrics.classification.MulticlassAccuracy(num_classes=wb.config['n_classes']).to(device=device),
        torchmetrics.classification.MulticlassPrecision(num_classes=wb.config['n_classes']).to(device=device),
        torchmetrics.classification.MulticlassRecall(num_classes=wb.config['n_classes']).to(device=device),
    ])
    
    # From https://docs.wandb.ai/guides/technical-faq/metrics-and-performance#can-i-log-metrics-on-two-different-time-scales-for-example-i-want-to-log-training-accuracy-per-batch-and-validation-accuracy-per-epoch
    # From https://docs.wandb.ai/guides/track/log/log-summary
    wb.define_metric("val_loss", summary="min")
    wb.define_metric("val_accuracy", summary="max")
    wb.define_metric("best_model_epoch")
    
    return metric_collection


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
        cfg,
        update_scheduler_on_batch: bool, 
        update_scheduler_on_epoch: bool,
        patience = None,
        val_loader = None,
        lr_milestones: List[int] = None,
    ): 

    train_start_time = time.time()
    total_batches_running = 0
    total_batches = len(loader)

    for epoch in range(start, n_epochs):
        model.train()
        loss_epoch = 0.0
        epoch_start_time = time.time()

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
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
        epoch_metrics = {}
        for metric in metrics.keys():
            epoch_metrics[metric] = float(metrics[metric].compute().cpu().data.numpy() * 100)
        metrics.reset()
        
        if update_scheduler_on_epoch:
            scheduler.step()
            if lr_milestones is not None:
                if epoch in lr_milestones:
                    logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
        
        logger.info('  Epoch {}/{}'.format(epoch + 1, n_epochs))
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

        # Validate on val set
        if val_loader:
            val_loss_norm, val_metrics, val_time, val_n_batches, _ = evaluate(val_loader, model, criterion, metrics, device, logger, validation=True)
            
            wb_log |= {
                'val_loss': val_loss_norm,
                'val_accuracy': val_metrics['MulticlassAccuracy'],
                'val_precision': val_metrics['MulticlassPrecision'],
                'val_recall': val_metrics['MulticlassRecall']
            }

            if patience:
                # Save best model
                if patience(val_metrics['MulticlassAccuracy']):
                    logger.info('  Found best model, saving model.')
                    best_model_epoch = epoch
                    
                    best_model_train_loss = epoch_loss_norm
                    best_model_train_accuracy = epoch_metrics['MulticlassAccuracy']
                    best_model_train_precision = epoch_metrics['MulticlassPrecision']
                    best_model_train_recall = epoch_metrics['MulticlassRecall']

                    best_model_val_loss = val_loss_norm
                    best_model_val_accuracy = val_metrics['MulticlassAccuracy']
                    best_model_val_precision = val_metrics['MulticlassPrecision']
                    best_model_val_recall = val_metrics['MulticlassRecall']

                    wb_log["best_model_epoch"] = best_model_epoch
                    best_checkpoint = {"state_dict": model.state_dict()}
                    
                    save_checkpoint(best_checkpoint, cfg['model_path'])

                checkpoint |= {
                    "max_accuracy": getattr(patience, 'baseline'),
                    "count": getattr(patience, 'count'),
                }

        wb.log(wb_log)            

        # Save checkpoint
        save_checkpoint(checkpoint, cfg['checkpoint_path'])
        
        # Early stopping
        if patience:
            if getattr(patience, 'count') == 0:
                logger.info('  Early stopping. Ending training.')
                break

    train_time = time.time() - train_start_time
    logger.info('Training time: %.3f' % train_time)

    train_results = {
        'total_batches': total_batches,
        'total_epochs': n_epochs,
        'train_time': train_time,
        'early_stopping': True if patience else False,
        'best_model_epoch': best_model_epoch if patience else epoch,
        'best_model_train_loss': best_model_train_loss if patience else epoch_loss_norm,
        'last_epoch_train_loss': epoch_loss_norm,
        'best_model_train_accuracy': best_model_train_accuracy if patience else epoch_metrics['MulticlassAccuracy'],
        'last_epoch_train_accuracy':  epoch_metrics['MulticlassAccuracy'],
        'best_model_train_precision': best_model_train_precision if patience else epoch_metrics['MulticlassPrecision'],
        'last_epoch_train_precision':  epoch_metrics['MulticlassPrecision'],
        'best_model_train_recall': best_model_train_recall if patience else epoch_metrics['MulticlassRecall'],
        'last_epoch_train_recall':  epoch_metrics['MulticlassRecall'],
    }
    if val_loader:
        train_results |= {
            'best_model_val_loss': best_model_val_loss if patience else val_loss_norm,
            'last_epoch_val_loss': val_loss_norm,
            'best_model_val_accuracy': best_model_val_accuracy if patience else val_metrics['MulticlassAccuracy'],
            'last_epoch_val_accuracy':  val_metrics['MulticlassAccuracy'],
            'best_model_val_precision': best_model_val_precision if patience else val_metrics['MulticlassPrecision'],
            'last_epoch_val_precision':  val_metrics['MulticlassPrecision'],
            'best_model_val_recall': best_model_val_recall if patience else val_metrics['MulticlassRecall'],
            'last_epoch_val_recall':  val_metrics['MulticlassRecall'],
        }

    return model, train_results


def evaluate(loader, model, criterion, metrics, device, logger, validation: bool):
    model.eval()
    running_loss = 0.0
    val_batches = len(loader)
    idx_label_scores = []
    start_time = time.time()

    log_str = 'Validation' if validation else 'Test'

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
    metric_dict = {}
    for metric in metrics.keys():
        metric_dict[metric] = float(metrics[metric].compute().cpu().data.numpy() * 100)
    metrics.reset()

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


class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


#### XAI
    # From https://github.com/utkuozbulak/pytorch-cnn-visualizations

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, output_path, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(output_path, file_name + '.png')
    save_image(gradient, path_to_file)


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()


    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)


    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr
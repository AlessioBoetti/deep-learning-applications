import os
import time
import copy
from typing import List
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from tqdm import tqdm
from datetime import datetime
from PIL import Image

import torch
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
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
    model = model.to(device)

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

        # Validate on val set
        if val_loader:
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

                checkpoint.update({
                    "max_accuracy": getattr(patience, 'baseline'),
                    "count": getattr(patience, 'count'),
                })

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
        'h-m-s_train_time': str(datetime.timedelta(seconds=train_time)),
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
        train_results.update({
            'best_model_val_loss': best_model_val_loss if patience else val_loss_norm,
            'last_epoch_val_loss': val_loss_norm,
            'best_model_val_accuracy': best_model_val_accuracy if patience else val_metrics['MulticlassAccuracy'],
            'last_epoch_val_accuracy':  val_metrics['MulticlassAccuracy'],
            'best_model_val_precision': best_model_val_precision if patience else val_metrics['MulticlassPrecision'],
            'last_epoch_val_precision':  val_metrics['MulticlassPrecision'],
            'best_model_val_recall': best_model_val_recall if patience else val_metrics['MulticlassRecall'],
            'last_epoch_val_recall':  val_metrics['MulticlassRecall'],
        })

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
    elif isinstance(im, torch.Tensor):
        im = T.functional.to_pil_image(im)
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
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_img, target_class):
        # Forward
        model_output = self.model(input_img)
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


def apply_colormap_on_image(org_img, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_img.size()[-2:])
    org_img = T.functional.to_pil_image(org_img) if isinstance(org_img, torch.Tensor) else org_img
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_img.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_class_activation_images(org_img, activation_map, filepath):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        filepath (str): File path of the exported image
    """
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    save_image(org_img, filepath + '_sample_image.png')
    save_image(heatmap, filepath + '_heatmap.png')
    save_image(heatmap_on_image, filepath + '_heatmap_on_image.png')
    save_image(activation_map, filepath + '_activation_grayscale.png')


class ClassActivationMapping_ORG:
    # From https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
    def __init__(self, model):
        self.model = model
        self.features_blobs = []
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()
    
    def hook_layers(self):
        def hook_function(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())

        # Register hook to the first layer
        last_layer = list(self.model.features._modules.items())[-1][1]   # Originally [0][1]
        last_layer.register_forward_hook(hook_function)
    
    def return_cam(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            # output_cam.append(cv2.resize(cam_img, size_upsample))
            output_cam.append(cam_img)
        return output_cam
    
    def generate_cam(self, input_img):
        params = list(self.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())
        logit = self.model(input_img)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        cams = self.return_cam(self.features_blobs[0], weight_softmax, [idx[0]])
        return cams


class ClassActivationMapping:
    # From https://arxiv.org/pdf/1512.04150.pdf
    # From https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.target_layer = target_layer
    
    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        # for module_pos, module in self.model._modules.items():
        #     x = module(x)
        #     if int(module_pos) == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer

        # for i, (module_name, module) in enumerate(self.model.named_modules()):
        #     print(f'Iteration {i}')
        #     print(module_name)
        #     x = module(x)
        #     print(f'Shape of x after module forward: {x.shape}')
        #     # if module_name == self.target_layer:
        #     if self.target_layer in module_name:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x

        x = self.model.conv_net(x)
        x.register_hook(self.save_gradient)
        conv_output = x
        return conv_output, x
    
    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x
    
    def generate_cam(self, input_img, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.forward_pass(input_img)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.conv_net.zero_grad()
        self.model.fc.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_img.shape[2],
                       input_img.shape[3]), Image.LANCZOS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_img[0].shape[1:])/np.array(cam.shape))
        return cam
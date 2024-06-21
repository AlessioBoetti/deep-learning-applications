import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


mnist_mean = (0.1307)
mnist_std = (0.3081)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)



def gradient_update(x, delta, alpha, epsilon):
    return (delta + x.shape[0]*alpha*delta.grad.data).clamp(-epsilon, epsilon)


def l_inf_update(delta, alpha, epsilon):
    grad = delta.grad.detach()
    return (delta + alpha*grad.sign()).clamp(-epsilon, epsilon)


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def l_2_update(x, delta, alpha, epsilon, lower_limit, upper_limit):
    delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
    delta.data = torch.clamp(delta.detach(), lower_limit - x, upper_limit - x)
    delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
    return delta.data


def criterion_fn(criterion, outputs, y):
    loss = criterion(outputs, y)
    return loss


def targeted_loss_ovo(outputs, y, y_targ):
    """
        Maximize the loss of the true label and minimize the loss of the alternative label.
    """
    loss = (outputs[:, y_targ] - outputs.gather(1, y[:, None])[:, 0]).sum()
    return loss


def targeted_loss_ovr(outputs, y_targ, y=None):
    """
        Maximize the target class logits and minimize the logits of all other classes.
    """
    loss = 2*outputs[:, y_targ].sum() - outputs.sum()
    return loss


def attack(model, x, y, epsilon: float, alpha: float, normalize: bool, criterion, scaler, device, logger, l_inf: bool = True, l_2: bool = False, dataset_name: str = None, fast: bool = True, target=None, restarts: int = 1, n_steps: int = 1, randomize: bool = True, mask = False):
    """
        If l_inf = True, restarts = 1, n_steps = 1 --> FGSM
        If restarts = 1, n_steps > 1               --> PGD
        If l_inf = True, restarts = 1, n_steps > 1 --> PGD with l_inf norm
        If l_2 = True, restarts = 1, n_steps > 1   --> PGD with l_2 norm
        
        If ..., randomize = True --> FGSM/PGD with randomization
        If ..., restarts > 1     --> FGSM/PGD with multiple restarts
        If ..., target           --> targeted FGSM/PGD (One-Vs-One or One-Vs-Rest)
        If ..., fast = True      --> Fast FGSM/PGD
        If ..., mask = True      --> Update delta batch value only for correctly classified examples


        References:
        - https://adversarial-ml-tutorial.org
        - https://github.com/locuslab/fast_adversarial
    """

    if fast:
        epsilon = 8 / 255.  # epsilon default value is 8 for CIFAR10 (from https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py)
        alpha = 10 / 255.  # alpha default value is 10 for CIFAR10 (from https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py)
        randomize = True

    if normalize:
        if dataset_name == 'mnist':
            mean = torch.tensor(mnist_mean, device=device).view(1, 1, 1)
            std = torch.tensor(mnist_std, device=device).view(1, 1, 1)
            input_dims = 1
        elif dataset_name == 'cifar10':
            mean = torch.tensor(cifar10_mean, device=device).view(3, 1, 1)
            std = torch.tensor(cifar10_std, device=device).view(3, 1, 1)
            input_dims = 3
        else:
            raise NotImplementedError()
        
        epsilon = epsilon / std
        alpha = alpha / std

        # used for scaling model input images into [0, 1] range
        lower_limit = ((0 - mean)/ std)
        upper_limit = ((1 - mean)/ std)
    else:
        lower_limit, upper_limit = 0, 1

    update_fn_kw = dict(alpha=alpha, epsilon=epsilon)
    if l_inf:
        update_fn = l_inf_update
    elif l_2:
        update_fn = l_2_update
        update_fn_kw.update({'x': x, 'lower_limit': lower_limit, 'upper_limit': upper_limit})
    else:
        update_fn = gradient_update
        update_fn_kw.update({'x': x})
    
    if not randomize and restarts > 1:
        logger.warning(f'PGD randomization was set to False, but restarts was set to more than 1. Setting randomization to True.')
        randomize = True

    if target:
        if target == 'OVO':  # faster but less consistent
            loss_fn = targeted_loss_ovo
        elif target == 'OVR':  # slower but more consistent
            loss_fn = targeted_loss_ovr
        loss_fn_kw = dict(y=y, y_targ=target)
    else:
        loss_fn = criterion_fn
        loss_fn_kw = dict(criterion=criterion, y=y)
    
    if restarts > 1:
        max_loss = torch.zeros(y.shape[0], device=device)
        max_delta = torch.zeros_like(x, device=device)

    for _ in range(restarts):
        delta = torch.zeros_like(x, requires_grad=True, device=device)
        if randomize:
            if fast:
                for dim in np.arange(input_dims):
                    delta[:, dim, :, :].uniform_(-epsilon[dim].item(), epsilon[dim].item())
            else:
                delta = torch.rand_like(x, requires_grad=True, device=device)
                delta.data = delta.data * 2 * epsilon - epsilon            
        
        delta.data = torch.clamp(delta.data, lower_limit - x, upper_limit - x)

        for _ in np.arange(n_steps):
            outputs = model(x + delta)  # outputs = model(x + delta[:x.size(0)])
            loss_fn_kw.update({'outputs': outputs})
            loss = loss_fn(**loss_fn_kw)
            scaler.scale(loss).backward()
            if mask:
                idx = outputs.argmax(1) == y
                delta.data[idx] = update_fn(delta=delta, **update_fn_kw)[idx]
                delta.data[idx] = torch.clamp(delta.data, lower_limit - x, upper_limit - x)[idx]
            delta.data = update_fn(delta=delta, **update_fn_kw)
            delta.data = torch.clamp(delta.data, lower_limit - x, upper_limit - x)
            delta.grad.zero_()
        
        if restarts > 1:
            outputs = model(x + delta)
            all_loss = F.cross_entropy(outputs, y, reduction='none')
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        else:
            max_loss = delta.detach()
    return max_loss


def draw_loss(model, x, y, criterion, epsilon, device):
    axis_range = np.linspace(-epsilon, epsilon, 100)
    Xi, Yi = np.meshgrid(axis_range, axis_range)
    
    def grad_at_delta(delta):
        delta.requires_grad_(True)
        loss = criterion(model(x + delta), y[0:1])
        loss.backward()
        return delta.grad.detach().sign().view(-1).cpu().numpy()

    dir1 = grad_at_delta(torch.zeros_like(x, requires_grad=True, device=device))
    delta2 = torch.zeros_like(x, requires_grad=True, device=device)
    delta2.data = torch.tensor(dir1).view_as(x).to(device)
    dir2 = grad_at_delta(delta2)
    np.random.seed(0)
    dir2 = np.sign(np.random.randn(dir1.shape[0]))
    
    all_deltas = torch.tensor((np.array([Xi.flatten(), Yi.flatten()]).T @ 
                              np.array([dir2, dir1])).astype(np.float32), device=device)
    outputs = model(all_deltas.view(-1, 1, 28, 28) + x)
    Zi = nn.CrossEntropyLoss(reduction="none")(outputs, y[0:1].repeat(yp.shape[0])).detach().cpu().numpy()
    Zi = Zi.reshape(*Xi.shape)
    #Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ls = LightSource(azdeg=0, altdeg=200)
    rgb = ls.shade(Zi, plt.cm.coolwarm)
    surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, linewidth=0, antialiased=True, facecolors=rgb)


class NormalizeInverse(T.Normalize):
    """
        Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


    def __call__(self, tensor):
        return super().__call__(tensor.clone())
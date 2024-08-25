import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


mnist_mean = (0.1307)
mnist_std = (0.3081)

cifar10_mean = (0.49139968, 0.48215827, 0.44653124)
cifar10_std = (0.24703233, 0.24348505, 0.26158768)

fashionmnist_mean = (0.485, 0.456, 0.406)  # coming from ImageNet values since it has millions of images
fashionmnist_std = (0.229, 0.224, 0.225)

svhn_mean = (0.485, 0.456, 0.406)  # coming from ImageNet values since it has millions of images
svhn_std = (0.229, 0.224, 0.225)


def gradient_update(delta, x, alpha, epsilon):
    return (delta + x.shape[0]*alpha*delta.grad.data).clamp(-epsilon, epsilon)


def l_inf_update(delta, alpha, epsilon):
    grad = delta.grad.detach()
    return (delta + alpha*grad.sign()).clamp(-epsilon, epsilon)


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def l_2_update(delta, x, alpha, epsilon, lower_limit, upper_limit):
    delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
    delta.data = torch.clamp(delta.detach(), lower_limit - x, upper_limit - x)
    delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
    return delta.data


def criterion_loss(outputs, y, criterion):
    """
        Maximize the loss of the true label.
    """
    loss = criterion(outputs, y)
    return loss


def targeted_loss(outputs, y_target, criterion, device):
    """
        Minimize the loss of the target label.
    """
    y_target = torch.tensor(y_target, device=device).unsqueeze(0).expand(outputs.shape[0])
    loss = -criterion(outputs, y_target)
    return loss


def targeted_loss_ovo(outputs, y, y_target, criterion, device):
    """
        Minimize the loss of the target label and maximize the loss of the true label.
        But since max(loss_target - loss_true) = max(output_true - output_target), we invert the signs, so we
        maximize the target class logits and minimize the true class logits. 
    """
    # loss = (outputs[:, y_target] - outputs.gather(1, y[:, None])[:, 0]).sum()
    y_target = torch.tensor(y_target, device=device).unsqueeze(0).expand(outputs.shape[0])
    loss = criterion(outputs, y) - criterion(outputs, y_target)
    return loss


def targeted_loss_ovr(outputs, y_target, criterion, n_classes, device):
    """
        Minimize the loss of the target label and maximize the loss of all other labels.
        But since max(loss_other - loss_true) = max(output_true - output_other), we invert the signs, so we
        maximize the target class logits and minimize the logits of all other classes.
    """
    # loss = 2*outputs[:, y_target].sum() - outputs.sum()

    y_target_t = torch.tensor(y_target, device=device).unsqueeze(0).expand(outputs.shape[0])
    loss = -criterion(outputs, y_target_t)
    for i in np.arange(n_classes):
        if i != y_target:
            i = torch.tensor(i, device=device).unsqueeze(0).expand(outputs.shape[0])
            loss += criterion(outputs, i)
    return loss


def attack(
    model, 
    x, 
    y, 
    epsilon: float, 
    alpha: float, 
    normalize: bool, 
    normalize_params: bool,
    criterion, 
    scaler, 
    device, 
    logger, 
    l_inf: bool = True, 
    l_2: bool = False, 
    dataset_name: str = None, 
    fast: bool = True, 
    fast_init: bool = True,
    target: int = None, 
    target_type: str = None, 
    restarts: int = 1, 
    steps: int = 1, 
    randomize: bool = True, 
    mask: bool = False, 
    until_success: bool = False,
    temperature: float = None,
    ):

    """
        If l_inf = True, restarts = 1, steps = 1 --> FGSM
        If restarts = 1, steps > 1               --> PGD
        If l_inf = True, restarts = 1, steps > 1 --> PGD with l_inf norm
        If l_2 = True, restarts = 1, steps > 1   --> PGD with l_2 norm
        
        If ..., randomize = True --> FGSM/PGD with randomization
        If ..., restarts > 1     --> FGSM/PGD with multiple restarts
        If ..., target           --> targeted FGSM/PGD (One-Vs-One or One-Vs-Rest)
        If ..., fast = True      --> Fast FGSM/PGD
        If ..., mask = True      --> Update delta batch value only for correctly classified examples


        References:
        - https://adversarial-ml-tutorial.org
        - https://github.com/locuslab/fast_adversarial

        Other references (unused):
        - https://adversarial-robustness-toolbox.readthedocs.io/en/latest/
        - https://github.com/AlbertMillan/adversarial-training-pytorch/blob/master/attacks.py
        - https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
        - https://pyimagesearch.com/2021/03/15/mixing-normal-images-and-adversarial-images-when-training-cnns/
        - https://github.com/LetheSec/Adversarial_Training_Pytorch/tree/main
    """

    """ 
        From https://adversarial-ml-tutorial.org/adversarial_examples/, section "The Fast Gradient Sign Method (FGSM)"
        FGSM is exactly the optimal attack against a linear binary classification model under the ℓ∞ norm. 
        This hopefully gives some additional helpful understanding of what FGSM is doing: 
        it assumes that the linear approximation of the hypothesis given by its gradient at the point x 
        is a reasonably good approximation to the function over the entire region $\|\delta\|\infty \leq \epsilon$. 
        It also, however, hints right away at the potential disadvantages to the FGSM attack: 
        because we know that neural networks are not in fact linear even over a relatively small region, 
        if we want a stronger attack we likely want to consider better methods at maximizing the loss function than a single projected gradient step. 
    """

    # From https://adversarial-ml-tutorial.org/introduction/, section "Training adversarially robust classifiers"
    # TODO: ... although in theory one can take just the worst-case perturbation as the point at which to compute the gradient, 
    # TODO: in practice this can cause osscilations of the training process, 
    # TODO: and it is often better to incorporate multiple perturbations with different random initializations and potentially also a gradient based upon the initial point with no perturbation.

    if fast:
        epsilon = 8 / 255.  # epsilon default value is 8 for CIFAR10 (from https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py)
        alpha = 10 / 255.  # alpha default value is 10 for CIFAR10 (from https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py)
        randomize = True
        fast_init = True
    else:
        epsilon = epsilon / 255.
        alpha = alpha / 255.

    dataset_name = dataset_name.lower().replace(' ', '')
    if dataset_name == 'mnist':
        n_classes = 10
        input_dims = 1
        mean = torch.tensor(mnist_mean, device=device).view(input_dims, 1, 1)
        std = torch.tensor(mnist_std, device=device).view(input_dims, 1, 1)
    elif dataset_name == 'cifar10':
        n_classes = 10
        input_dims = 3
        mean = torch.tensor(cifar10_mean, device=device).view(input_dims, 1, 1)
        std = torch.tensor(cifar10_std, device=device).view(input_dims, 1, 1)
    elif dataset_name == 'fashionmnist':
        n_classes = 10
        input_dims = 1
        mean = torch.tensor(fashionmnist_mean, device=device).view(input_dims, 1, 1)
        std = torch.tensor(fashionmnist_std, device=device).view(input_dims, 1, 1)
    elif dataset_name == 'svhn':
        n_classes = 10
        input_dims = 3
        mean = torch.tensor(svhn_mean, device=device).view(input_dims, 1, 1)
        std = torch.tensor(svhn_std, device=device).view(input_dims, 1, 1)
    else:
        raise NotImplementedError()
    
    if normalize:
        lower_limit = ((0 - mean)/ std)
        upper_limit = ((1 - mean)/ std)
    else:
        lower_limit, upper_limit = 0, 1

    if normalize and normalize_params:
        epsilon = epsilon / std
        alpha = alpha / std
    else:
        epsilon = epsilon / torch.ones_like(std)
        alpha = alpha / torch.ones_like(std)

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
        logger.warning(f'Randomization was set to False, but restarts was set to more than 1. Setting randomization to True.')
        randomize = True

    if target:
        loss_fn_kw = dict(y_target=target, criterion=criterion, device=device)
        if target_type == 'OVO':  # faster but less consistent
            loss_fn = targeted_loss_ovo
            loss_fn_kw.update({'y': y})
        elif target_type == 'OVR':  # slower but more consistent
            loss_fn = targeted_loss_ovr
            loss_fn_kw.update({'n_classes': n_classes})
        else:
            loss_fn = targeted_loss
    else:
        loss_fn = criterion_loss
        loss_fn_kw = dict(y=y, criterion=criterion)
    
    if restarts > 1:
        max_loss = torch.zeros(y.shape[0], device=device)
        max_delta = torch.zeros_like(x, device=device)

    for _ in range(restarts):
        delta = torch.zeros_like(x, device=device)
        if randomize:
            if fast_init:
                for dim in np.arange(input_dims):
                    delta[:, dim, :, :] = delta[:, dim, :, :].uniform_(-epsilon[dim].item(), epsilon[dim].item())
            else:
                delta = torch.rand_like(x, device=device)
                delta.data = delta.data * 2 * epsilon - epsilon
        delta.requires_grad_(True)
        
        delta.data = torch.clamp(delta.data, lower_limit - x, upper_limit - x)

        if until_success:
            outputs = model(x + delta)  # outputs = model(x + delta[:x.size(0)])
            if temperature:
                outputs = outputs / temperature
            while (outputs.argmax(1) == y).any():
                loss_fn_kw.update({'outputs': outputs})
                loss = loss_fn(**loss_fn_kw)
                scaler.scale(loss).backward()
                idx = outputs.argmax(1) == y
                delta.data[idx] = update_fn(delta, **update_fn_kw)[idx]
                delta.data[idx] = torch.clamp(delta.data, lower_limit - x, upper_limit - x)[idx]
                delta.grad.zero_()
                outputs = model(x + delta)  # outputs = model(x + delta[:x.size(0)])
                if temperature:
                    outputs = outputs / temperature
        else:
            for _ in np.arange(steps):
                outputs = model(x + delta)  # outputs = model(x + delta[:x.size(0)])
                if temperature:
                    outputs = outputs / temperature
                loss_fn_kw.update({'outputs': outputs})
                loss = loss_fn(**loss_fn_kw)
                scaler.scale(loss).backward()
                if mask:
                    idx = outputs.argmax(1) == y
                    delta.data[idx] = update_fn(delta, **update_fn_kw)[idx]
                    delta.data[idx] = torch.clamp(delta.data, lower_limit - x, upper_limit - x)[idx]
                else:
                    delta.data = update_fn(delta, **update_fn_kw)
                    delta.data = torch.clamp(delta.data, lower_limit - x, upper_limit - x)
                delta.grad.zero_()
        
        if restarts > 1:
            outputs = model(x + delta)
            if temperature:
                outputs = outputs / temperature
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
    # Zi = (Zi-Zi.min())/(Zi.max() - Zi.min())
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ls = LightSource(azdeg=0, altdeg=200)
    rgb = ls.shade(Zi, plt.cm.coolwarm)
    surf = ax.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, linewidth=0, antialiased=True, facecolors=rgb)


def plot_images(inputs, labels, outputs, M, N, out_path, classes, adv: bool = False, alpha: int = None, diff: bool = False, n: str = ''):
    
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(25, 25))
    for i in range(M):
        for j in range(N):
            if inputs.shape[1] > 1:
                img = inputs[i*N+j].permute(1, 2, 0).detach().cpu().numpy()
                ax[i][j].imshow((img * 255).astype(np.uint8))
            else:
                img = inputs[i*N+j][0].detach().cpu().numpy()
                ax[i][j].imshow(img, cmap="gray")
            if alpha and not diff:
                title_string = "Pred: {}".format(classes[outputs[i*N+j].max(dim=0)[1]])
                title = ax[i][j].set_title(title_string, size=20)
                plt.setp(title, color=('g' if outputs[i*N+j].max(dim=0)[1] == labels[i*N+j] else 'r'))
            else:
                title_string = "True: {}".format(classes[labels[i*N+j]])
                title = ax[i][j].set_title(title_string, size=20)
            ax[i][j].set_axis_off()
    if alpha and not diff:
        suptitle = f"Corrupted images - alpha (eps): {alpha}"
    elif diff:
        suptitle = "Diffs"
    else:
        suptitle = 'Original images'
    plt.suptitle(suptitle, size=20)
    plt.tight_layout()
    if adv:
        plt.savefig(f'{out_path}/corrupted_images{n}.png')
    elif diff:
        plt.savefig(f'{out_path}/diffs{n}.png')
    else:
        plt.savefig(f'{out_path}/sample_images{n}.png')
    plt.close()


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
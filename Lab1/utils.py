import torch
import torchmetrics
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as T
import os
from torchvision.datasets import CIFAR10


def save_checkpoint(string, state, directory):
    print(string)
    torch.save(state, "".join([directory, "model.pth.tar"]))


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['start'], checkpoint['max_accuracy'], checkpoint['count']


def create_directory_if_does_not_exist(check_dir, w_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)


def load_best_model(checkpoint, model):
    print("Loading best model")
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(batch_size, num_workers, training=True):
    transform = T.Compose([
        T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    test_ds = CIFAR10(
        root='./Dataset/',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
    )

    if training:
        train_ds = CIFAR10(
            root='./Dataset/',
            train=True,
            download=True
            transform=transform
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )
        return train_loader, test_loader
    
    return test_loader


def metrics(wb, device):
    metric_collection = torchmetrics.MetricCollection([
        torchmetrics.classification.MulticlassAccuracy(num_classes=wb.config['num_class']).to(device=device)
    ])
    wb.define_metric("test_loss", summary="min")
    wb.define_metric("test_accuracy", summary="max")
    wb.define_metric("accuracy_epoch")
    return metric_collection


def eval_fn(loader, model, criterion, metric_collection, device):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(data)
            loss = criterion(prediction, target)
            running_loss += loss.item()
            metric_collection(prediction, target)

    loss = running_loss / len(loader)
    accuracy = metric_collection['MulticlassAccurcay'].compute().cpu() * 100

    print(f"Got on test set --> Accuracy: {accuracy:.3f} and Loss: {loss:.3f}")

    metric_collection.reset()
    return loss, accuracy


def save_plot(train_l, train_a, test_l, test_a):
    plt.plot(train_a, "-")
    plt.plot(test_a, "-")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["Train", "Valid"])
    plt.title("Train Vs. Valid Accuracy")
    plt.savefig('result/accuracy')
    plt.close()

    plt.plot(train_l, "-")
    plt.plot(test_l, "-")
    plt.xlabel("epoch")
    plt.ylabel("losses")
    plt.legend(["Train", "Valid"])
    plt.title("Train Vs. Valid Losses")
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

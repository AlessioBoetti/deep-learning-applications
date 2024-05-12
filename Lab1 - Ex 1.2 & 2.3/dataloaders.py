import os
import random
import numpy as np
from typing import Union, Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, DatasetFolder


def get_target_label_idx(labels, targets):
    """
        Get the indices of labels that are included in targets.
        :param labels: array of labels
        :param targets: list/tuple of target labels
        :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
        Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
        which is either the standard deviation, L1- or L2-norm across features (pixels).
        Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        """
            Override the original method of the MNIST class.
            Args:
                index (int): Index
            Returns:
                triple: (image, target, index) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class BaseDataset(object):
    def __init__(self):
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.org_test_set = None
        self.train_sampler = None
        self.val_sampler = None
        self.use_sampler = None

    def __repr__(self):
        return self.__class__.__name__
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def loaders(
        self, 
        train: bool,  
        batch_size: int,
        val: bool = False, 
        shuffle_train: bool = False, 
        num_workers: int = 0, 
        pin_memory: bool = False, 
        persistent_workers: bool = False,
        seed: int = 1,
    ) -> Union[DataLoader, DataLoader]:
                
        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        g = torch.Generator()
        g.manual_seed(seed)
        kw = dict(
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            persistent_workers=persistent_workers,
            worker_init_fn=self.seed_worker,
            generator=g
        )

        # Shuffling the dataset is only needed for train set.
        # Shuffling should be done every epoch (shuffle=True does this).

        test_loader = DataLoader(dataset=self.test_set, **kw)
        org_test_loader = DataLoader(dataset=self.org_test_set, **kw)
        if train:
            shuffle_train = False if self.train_sampler is not None else True
            train_loader = DataLoader(dataset=self.train_set, shuffle=shuffle_train, sampler=self.train_sampler, **kw)
            if val:
                shuffle_val = False if self.val_sampler is not None else shuffle_val
                val_loader = DataLoader(dataset=self.val_set, sampler=self.val_sampler, **kw)
                return train_loader, val_loader, test_loader, org_test_loader
            else:
                return train_loader, None, test_loader, org_test_loader        
        return None, None, test_loader, org_test_loader


class MNIST_Dataset(BaseDataset):
    def __init__(
        self, 
        root: str,
        val_size: float = None,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        problem: str = None, 
        normal_class: int = None, 
        multiclass: bool = None, 
        img_size: int = None,
        normalize: bool = False,
        gcn: str = None,   # 'l1' oppure 'l2' 
        gcn_minmax: bool = False,
        use_sampler: bool = True,
    ):
        super().__init__()
        self.root = root

        if problem is not None:
            if problem.lower() == 'od':
                self.n_classes = 2  # 0: normal, 1: outlier
                self.normal_classes = tuple([normal_class])  # tuple with original class labels that define the normal class
                self.outlier_classes = list(range(0, 10))  # tuple with original class labels that define the outlier class
                self.outlier_classes.remove(normal_class)
            else:
                raise NotImplementedError(f'Problem {problem} is not valid. Only Outlier Detection is implemented.')

        if gcn:
            # Pre-computed min and max values (after applying GCN) from train data per class
            min_max = [(-0.8826567065619495, 9.001545489292527),
                    (-0.6661464580883915, 20.108062262467364),
                    (-0.7820454743183202, 11.665100841080346),
                    (-0.7645772083211267, 12.895051191467457),
                    (-0.7253923114302238, 12.683235701611533),
                    (-0.7698501867861425, 13.103278415430502),
                    (-0.778418217980696, 10.457837397569108),
                    (-0.7129780970522351, 12.057777597673047),
                    (-0.8280402650205075, 10.581538445782988),
                    (-0.7369959242164307, 10.697039838804978)]

        transform = transforms.Compose([
            transforms.Resize(img_size) if img_size else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if normalize else lambda x: x,
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale=gcn)) if gcn else lambda x: x,
            transforms.Normalize([min_max[normal_class][0]],
                                 [min_max[normal_class][1] - min_max[normal_class][0]]) if gcn_minmax else lambda x: x,
        ])
        
        if problem is not None:
            if problem.lower() == 'od':
                target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        else:
            target_transform = None

        self.train_set = MyMNIST(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
        self.test_set = MyMNIST(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)

        if problem is not None:
            if problem.lower() == 'od':
                train_idx_normal = get_target_label_idx(self.train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
                self.train_set = Subset(self.train_set, train_idx_normal)
            else:
                raise NotImplementedError(f'Problem {problem} is not valid. Only Outlier Detection is implemented.')
        
        if val_size:
            self.val_set = MyMNIST(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            n_train = len(self.train_set)
            indices = list(range(n_train))
            split = int(np.floor(val_size * n_train))
            if val_shuffle:
                np.random.seed(val_shuffle_seed)
                np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split] 
            
            if use_sampler:
                self.train_sampler = SubsetRandomSampler(train_idx)
                self.val_sampler = SubsetRandomSampler(val_idx)
            else:
                self.train_set = Subset(self.train_set, train_idx)
                self.val_set = Subset(self.val_set, val_idx)


class CIFAR10_Dataset(BaseDataset):
    def __init__(self, 
        root: str,
        val_size: float = None,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        problem: str = None, 
        normal_class: int = None, 
        multiclass: bool = None, 
        img_size: int = None,
        normalize: bool = False,
        gcn: bool = False, 
        gcn_minmax: bool = False,
        augment: bool = False,
        use_sampler: bool = True,
    ):
        super().__init__()
        self.root = root

        if problem is not None:
            if problem.lower() == 'od':
                self.n_classes = 2  # 0: normal, 1: outlier
                self.normal_classes = tuple([normal_class])  # tuple with original class labels that define the normal class
                self.outlier_classes = list(range(0, 10))  # tuple with original class labels that define the outlier class
                self.outlier_classes.remove(normal_class)
            else:
                raise NotImplementedError(f'Problem {problem} is not valid. Only Outlier Detection is implemented.')

        if gcn:
            # Pre-computed min and max values (after applying GCN) from train data per class
            min_max = [(-28.94083453598571, 13.802961825439636),
                    (-6.681770233365245, 9.158067708230273),
                    (-34.924463588638204, 14.419298165027628),
                    (-10.599172931391799, 11.093187820377565),
                    (-11.945022995801637, 10.628045447867583),
                    (-9.691969487694928, 8.948326776180823),
                    (-9.174940012342555, 13.847014686472365),
                    (-6.876682005899029, 12.282371383343161),
                    (-15.603507135507172, 15.2464923804279),
                    (-6.132882973622672, 8.046098172351265)]
            
        transform = transforms.Compose([
            transforms.Resize(img_size) if img_size else lambda x: x,
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10) if augment else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else lambda x: x,       # Correct values are: mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233 0.24348505 0.26158768)
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')) if gcn else lambda x: x,
            transforms.Normalize([min_max[normal_class][0]] * 3,
                                 [min_max[normal_class][1] - min_max[normal_class][0]] * 3) if gcn_minmax else lambda x: x,
        ])

        basic_transform = transforms.Compose([
            transforms.Resize(img_size) if img_size else lambda x: x,
            transforms.ToTensor()
        ])

        if problem is not None:
            if problem.lower() == 'od':
                target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        else:
            target_transform = None

        self.train_set = MyCIFAR10(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
        self.test_set = MyCIFAR10(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
        self.org_test_set = MyCIFAR10(root=self.root, train=False, download=True, transform=basic_transform, target_transform=target_transform)

        if problem is not None:
            if problem.lower() == 'od':
                train_idx_normal = get_target_label_idx(self.train_set.train_labels, self.normal_classes)
                self.train_set = Subset(self.train_set, train_idx_normal)

        if val_size:
            self.val_set = MyCIFAR10(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            n_train = len(self.train_set)
            indices = list(range(n_train))
            split = int(np.floor(val_size * n_train))
            if val_shuffle:
                np.random.seed(val_shuffle_seed)
                np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            
            if use_sampler:
                self.train_sampler = SubsetRandomSampler(train_idx)
                self.val_sampler = SubsetRandomSampler(val_idx)     # Shuffling val set is not necessary
            else:
                self.train_set = Subset(self.train_set, train_idx)
                self.val_set = Subset(self.val_set, val_idx)
        
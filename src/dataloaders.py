import os
import random
import numpy as np
import pandas as pd
from typing import Union, Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN, ImageFolder, DatasetFolder

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from datasets import load_dataset


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        self.problem = None
        self.n_classes = None
        self.normal_classes = None
        self.outlier_classes = None


    def __repr__(self):
        return self.__class__.__name__


    def loaders(
        self, 
        train: bool,  
        batch_size: int,
        device: str,
        val: bool = False, 
        shuffle_train: bool = False, 
        num_workers: int = 0, 
        pin_memory: bool = False, 
        persistent_workers: bool = False,
        seed: int = 1,
        ) -> Union[DataLoader, DataLoader]:
                
        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        g = torch.Generator(device)
        g.manual_seed(seed)
        kw = dict(
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
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
            return train_loader, None, test_loader, org_test_loader        
        return None, None, test_loader, org_test_loader


    def _setup_in_out_classes(self, n_classes, normal_class, total_classes):
        self.n_classes = n_classes  # if n_classes = 2, then 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])  # tuple with original class labels that define the normal class
        self.outlier_classes = list(range(0, total_classes))  # tuple with original class labels that define the outlier class
        self.outlier_classes.remove(normal_class)


    def _get_transforms(
        self, 
        img_size: bool = None,
        augment: str = None,
        normalize = None, 
        gcn: str = None, 
        gcn_minmax: bool = None, 
        min_max = None, 
        normal_class: int = None, 
        gcn_dims: int = 1
        ):
        
        transform_list, basic_transform_list = [], []
        if img_size:
            transform_list.append(transforms.Resize(img_size))
            basic_transform_list.append(transforms.Resize(img_size))
        if augment:
            if augment == 'CIFAR10':
                policy = transforms.AutoAugmentPolicy.CIFAR10
            elif augment == 'IMAGENET':
                policy = transforms.AutoAugmentPolicy.IMAGENET
            elif augment == 'SVHN':
                policy = transforms.AutoAugmentPolicy.SVHN
            else:
                raise ValueError(f'AutoAugmentPolicy type not supported by torchvision.')
            transform_list.append(transforms.AutoAugment(policy))
        transform_list.append(transforms.ToTensor())
        basic_transform_list.append(transforms.ToTensor())
        if normalize:
            transform_list.append(transforms.Normalize(normalize[0], normalize[1]))
        if gcn:
            transform_list.append(transforms.Lambda(lambda x: global_contrast_normalization(x, scale=gcn)))
        if gcn_minmax:
            transform_list.append(transforms.Normalize(
                [min_max[normal_class][0]] * gcn_dims,
                [min_max[normal_class][1] - min_max[normal_class][0]] * gcn_dims
            ))
        
        transform = transforms.Compose(transform_list)
        basic_transform = transforms.Compose(basic_transform_list)

        # When using PyTorch with CUDA, composing transforms where at least one transform is a lambda (no transform) causes problem when parallelizing processes,
        # so we can't use the following code and we need to revert to the above code
        
        """ 
            transform = transforms.Compose([
                transforms.Resize(img_size) if img_size else lambda x: x,
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) if normalize else lambda x: x,
                transforms.Lambda(lambda x: global_contrast_normalization(x, scale=gcn)) if gcn else lambda x: x,
                transforms.Normalize([min_max[normal_class][0]],
                                    [min_max[normal_class][1] - min_max[normal_class][0]]) if gcn_minmax else lambda x: x,
            ])

            basic_transform = transforms.Compose([
                transforms.Resize(img_size) if img_size else lambda x: x,
                transforms.ToTensor()
            ]) 
        """
        
        target_transform = None
        if self.problem is not None:
            if self.problem == 'ood':
                if self.outlier_classes is not None:
                    target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        
        return transform, basic_transform, target_transform


    def _setup_splits(
        self,
        MyDataset,
        transform, 
        basic_transform, 
        target_transform, 
        val_size: float = None, 
        val_mix: bool = None,  # Not implemented
        val_shuffle: bool = None, 
        val_shuffle_seed: int = None,
        use_sampler: bool = None,
        is_svhn: bool = False,
        ):

        if is_svhn:
            self.train_set = MyDataset(root=self.root, split='train', download=True, transform=transform, target_transform=target_transform)
            self.test_set = MyDataset(root=self.root, split='test', download=True, transform=transform, target_transform=target_transform)
            self.org_test_set = MyDataset(root=self.root, split='test', download=True, transform=basic_transform, target_transform=target_transform)
        else:
            self.train_set = MyDataset(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            self.test_set = MyDataset(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            self.org_test_set = MyDataset(root=self.root, train=False, download=True, transform=basic_transform, target_transform=target_transform)

        if self.problem is not None:
            if self.problem == 'ood':
                if self.normal_classes is not None:
                    self.train_idx_normal = get_target_label_idx(self.train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
                    self.train_set = Subset(self.train_set, self.train_idx_normal)
            
        if val_size:
            if is_svhn:
                self.val_set = MyDataset(root=self.root, split='train', download=True, transform=transform, target_transform=target_transform)
            else:
                self.val_set = MyDataset(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            # TODO: If we are doing OD, the train set may not have OOD data. If this is the case, to include OOD data in the val set
            # we should select n_samples from val_set, but if val_set is a copy of the full train_set, n_samples is bigger than reduced train_set!
            n_samples = len(self.train_set)
            indices = list(range(n_samples))
            split = int(np.floor(val_size * n_samples))
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


class MyFashionMNIST(FashionMNIST):
    """Torchvision FashionMNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        """
            Override the original method of the FashionMNIST class.
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


class MySVHN(SVHN):
    """Torchvision SVHN class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        """
            Override the original method of the SVHN class.
            Args:
                index (int): Index
            Returns:
                triple: (image, target, index) where target is index of the target class.
        """
        
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


class MNIST_Dataset(BaseDataset):
    def __init__(
        self, 
        root: str,
        problem: str = None, 
        n_classes: int = 2,
        normal_class: Union[int, List[int]] = None,
        total_classes: int = 10,
        img_size: int = None,
        normalize: bool = False,
        gcn: str = None,   # 'l1' oppure 'l2'
        gcn_minmax: bool = False,
        augment: bool = False,
        val_size: float = None,
        val_mix: bool = False,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        use_sampler: bool = True,
        multiclass: bool = None,  # TODO: Implement from DL PW
        ):
        
        super().__init__()
        self.root = root
        self.norm_stats = [(0.1307,), (0.3081,)]
        
        if problem is not None:
            self.problem = problem.lower().replace(' ', '')
            if self.problem == 'ood':
                if normal_class is not None:
                    self._setup_in_out_classes(n_classes, normal_class, total_classes)
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
        else:
            min_max = None

        transform, basic_transform, target_transform = self._get_transforms(
            img_size=img_size, 
            augment=False, 
            normalize=self.norm_stats if normalize else False, 
            gcn=gcn, 
            gcn_minmax=gcn_minmax, 
            min_max=min_max,
            normal_class=normal_class,
        )
        
        self._setup_splits(
            MyMNIST, 
            transform, 
            basic_transform, 
            target_transform, 
            val_size, 
            val_mix, 
            val_shuffle, 
            val_shuffle_seed, 
            use_sampler,
        )


class CIFAR10_Dataset(BaseDataset):
    def __init__(
        self, 
        root: str,
        problem: str = None,
        n_classes: int = 2,
        normal_class: Union[int, List[int]] = None,
        total_classes: int = 10,
        img_size: int = None,
        normalize: bool = False,
        gcn: str = None,   # 'l1' oppure 'l2'
        gcn_minmax: bool = False,
        augment: bool = False,
        val_size: float = None,
        val_mix: bool = False,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        use_sampler: bool = True,
        multiclass: bool = None,  # TODO: Implement from DL PW
        ):

        super().__init__()
        self.root = root
        self.norm_stats = [(0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)]

        if problem is not None:
            self.problem = problem.lower().replace(' ', '')
            if self.problem == 'ood':
                if normal_class is not None:
                    self._setup_in_out_classes(n_classes, normal_class, total_classes)
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
        else:
            min_max = None

        transform, basic_transform, target_transform = self._get_transforms(
            img_size=img_size, 
            augment='CIFAR10' if augment else False, 
            normalize=self.norm_stats if normalize else False, 
            gcn=gcn, 
            gcn_minmax=gcn_minmax, 
            min_max=min_max,
            normal_class=normal_class,
            gcn_dims=3, 
        )

        self._setup_splits(
            MyCIFAR10, 
            transform, 
            basic_transform, 
            target_transform, 
            val_size, 
            val_mix, 
            val_shuffle, 
            val_shuffle_seed, 
            use_sampler,
        )


class FashionMNIST_Dataset(BaseDataset):
    def __init__(
        self, 
        root: str,
        problem: str = None,
        n_classes: int = 2,
        normal_class: Union[int, List[int]] = None,
        total_classes: int = 10,
        img_size: int = None,
        normalize: bool = False,
        gcn: str = None,   # 'l1' oppure 'l2'
        gcn_minmax: bool = False,
        augment: bool = False,
        val_size: float = None,
        val_mix: bool = False,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        use_sampler: bool = True,
        multiclass: bool = None,  # TODO: Implement from DL PW
        ):
        
        super().__init__()
        self.root = root
        self.norm_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]  # coming from ImageNet values since it has millions of images

        if problem is not None:
            self.problem = problem.lower().replace(' ', '')
            if self.problem == 'ood':
                if normal_class is not None:
                    self._setup_in_out_classes(n_classes, normal_class, total_classes)
            else:
                raise NotImplementedError(f'Problem {problem} is not valid. Only Outlier Detection is implemented.')

        if gcn:
            # Pre-computed min and max values (after applying GCN) from train data per class
            min_max = None
        else:
            min_max = None
        
        transform, basic_transform, target_transform = self._get_transforms(
            img_size=img_size, 
            augment=False, 
            normalize=self.norm_stats if normalize else False, 
            gcn=gcn, 
            gcn_minmax=gcn_minmax, 
            normal_class=normal_class,
        )

        self._setup_splits(
            MyFashionMNIST, 
            transform, 
            basic_transform, 
            target_transform, 
            val_size, 
            val_mix, 
            val_shuffle, 
            val_shuffle_seed, 
            use_sampler,
        )


class SVHN_Dataset(BaseDataset):
    def __init__(
        self, 
        root: str,
        problem: str = None,
        n_classes: int = 2,
        normal_class: Union[int, List[int]] = None,
        total_classes: int = 10,
        img_size: int = None,
        normalize: bool = False,
        gcn: str = None,   # 'l1' oppure 'l2'
        gcn_minmax: bool = False,
        augment: bool = False,
        val_size: float = None,
        val_mix: bool = False,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        use_sampler: bool = True,
        multiclass: bool = None,  # TODO: Implement from DL PW
    ):
        super().__init__()
        self.root = root
        self.norm_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]  # coming from ImageNet values since it has millions of images

        if problem is not None:
            self.problem = problem.lower().replace(' ', '')
            if self.problem == 'ood':
                if normal_class is not None:
                    self._setup_in_out_classes(n_classes, normal_class, total_classes)
            else:
                raise NotImplementedError(f'Problem {problem} is not valid. Only Outlier Detection is implemented.')

        if gcn:
            # Pre-computed min and max values (after applying GCN) from train data per class
            min_max = None
        else:
            min_max = None
        
        transform, basic_transform, target_transform = self._get_transforms(
            img_size=img_size, 
            augment=False, 
            normalize=self.norm_stats if normalize else False, 
            gcn=gcn, 
            gcn_minmax=gcn_minmax, 
            normal_class=normal_class,
        )

        self._setup_splits(
            MySVHN, 
            transform, 
            basic_transform, 
            target_transform, 
            val_size, 
            val_mix, 
            val_shuffle, 
            val_shuffle_seed, 
            use_sampler,
            is_svhn=True,
        )


class QADataset(Dataset):
    def __init__(self) -> None:
        ...


class TextClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, data_dir, model_name, cache_dir, padding_side, trunc_side, device, max_token_len=128, filename=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = padding_side
        self.tokenizer.truncation_side = trunc_side
        self.max_token_len = max_token_len
        # self.device = device

        self.text, self.label = [], []
        self.load_dataset(dataset_name, split, data_dir, filename)


    def load_dataset(self, dataset_name, split, data_dir, filename=None):
        csv_file = f'{dataset_name}_{split}.csv'
        if csv_file in os.listdir(data_dir):
            df = pd.read_csv(f'{data_dir}/{csv_file}')
        else:
            if filename:
                hf_ds_kw = dict(repo_id=dataset_name, filename=filename, repo_type='dataset', local_dir=data_dir, local_dir_use_symlinks=False)
                if filename.endswith('.parquet'):
                    df = pd.read_parquet(hf_hub_download(**hf_ds_kw))
                elif filename.endswith('.csv'):
                    df = pd.read_csv(hf_hub_download(**hf_ds_kw))
                else:
                    raise NotImplementedError('Dataset filename is a not implemented file type.')
            else:
                df = load_dataset(dataset_name, split=split, cache_dir=data_dir)
                df = df.to_pandas()
                dataset_name = dataset_name.replace('/', '_')
                df.to_csv(f'{data_dir}/{dataset_name}_{split}.csv', index=False)
        
        df = df.fillna('')
        df = df.astype(str)
        df['text'] = df['text'].str.lower().replace(r'\n', ' ')
        df['label'] = df['label'].astype(int)

        for idx in df.index:
            row = df.loc[idx]
            self.text.append(row['text'])
            self.label.append(row['label'])   


    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        text = self.text[index]
        label = self.label[index]
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_token_len,
            return_attention_mask=True
        ) # .to(self.device)
        return tokens.input_ids.flatten(), tokens.attention_mask.flatten(), label
    

class NLP_Dataset(BaseDataset):
    def __init__(self, 
        dataset_type: str,
        dataset_name: str,
        data_dir: str,
        train_set_name: str,
        test_set_name: str,
        model_name: str,
        cache_dir: str, 
        padding_side, 
        trunc_side,
        max_token_len: int,
        device: str,
        filename: str = None,
        val_set_name: bool = False,
        val_size: float = None,
        val_shuffle: bool = True,
        val_shuffle_seed: int = 1,
        use_sampler: bool = True,
    ):   
        super().__init__()
        dataset_kw = dict(
            dataset_name=dataset_name,
            filename=filename,
            data_dir=data_dir,
            model_name=model_name,
            cache_dir=cache_dir, 
            padding_side=padding_side, 
            trunc_side=trunc_side, 
            max_token_len=max_token_len,
            device=device, 
        )
        if dataset_type.lower().replace(' ', '') == 'textclassification':
            self.train_set = TextClassificationDataset(split=train_set_name, **dataset_kw)
            self.test_set = TextClassificationDataset(split=test_set_name, **dataset_kw)
            if val_set_name:
                self.val_set = TextClassificationDataset(split=val_set_name, **dataset_kw)
            elif val_size:
                self.val_set = TextClassificationDataset(split=train_set_name, **dataset_kw)                
                
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
        else:
            raise NotImplementedError(f'Dataset type {dataset_type} not implemented.')
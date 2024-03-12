from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


""" class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = None
        self.net_name = None

    def load_model(self, model_path, load_pretrainer: bool = False):
        model_dict = torch.load(model_path)
        self.net.load_state_dict(model_dict['net_dict'])
        if load_pretrainer:
            if self.pretrainer_net is None:
                self.pretrainer_net = self.build_pretrainer(self.net_name)
            self.pretrainer_net.load_state_dict(model_dict['pretrainer_net_dict'])
    

    def save_model(self, model_path, save_pretrainer: bool = False):
        net_dict = self.net.state_dict()
        pretrainer_net_dict = self.pretrainer_net.state_dict() if save_pretrainer else None
        torch.save({'net_dict': net_dict, 'pretrainer_net_dict': pretrainer_net_dict}, model_path)
    

    def build_pretrainer(self):
        pass """


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise NotImplementedError("Activation function '{}' is not implemented.".format(self.activation))
    
    def _get_pooling(self, pooling, kw):
        if pooling.lower() == 'adaptivemaxpool':
            return nn.AdaptiveMaxPool2d(**kw)
        elif pooling.lower() == 'maxpool':
            return nn.MaxPool2d(**kw)
        else:
            raise NotImplementedError("Activation function '{}' is not implemented.".format(pooling))


@torch.no_grad()
def init_weights(m):
    # From https://pytorch.org/docs/stable/notes/modules.html#modules-as-building-blocks
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.fill_(0.0)


class MultiLayerPerceptron(BaseModel):
    def __init__(
            self, 
            n_hidden_layers: int, 
            input_size: int, 
            hidden_layer_sizes: List[int], 
            output_size: int, 
            activation: str, 
            batch_norm: bool = False, 
            dropout_prob: float = None,
            last_activation: str = None,
            flatten_input: bool = False,
        ):
        super().__init__()
        
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = True if dropout_prob else False
        self.dropout_prob = dropout_prob
        self.last_activation = last_activation
        self.flatten_input = flatten_input

        bias = False if batch_norm else True
        
        assert n_hidden_layers == len(hidden_layer_sizes), "Number of hidden layers doesn't match with given hidden sizes."
    
        self.layers = nn.Sequential()
        in_dim = input_size
        for i, hidden_dim in enumerate(hidden_layer_sizes):
            self.layers.add_module(f'linear_{i+1}', nn.Linear(in_dim, hidden_dim, bias))
            if batch_norm:
                self.layers.add_module(f'bn_{i+1}', nn.BatchNorm1d(hidden_dim))
            self.layers.add_module(f'act_{i+1}', self._get_activation(activation))
            if dropout_prob:
                self.layers.add_module(f'dropout_{i+1}', nn.Dropout(dropout_prob))
            in_dim = hidden_dim
        self.layers.add_module(f'last_linear', nn.Linear(hidden_layer_sizes[-1], output_size, bias))
        if last_activation:
            self.layers.add_module('last_act', self._get_activation(last_activation))

        """ self.layers = nn.Sequential()
        self.layers.add_module('linear_0', nn.Linear(input_size, hidden_layer_sizes[0], bias))
        for i in range(n_hidden_layers - 1):
            self.layers.add_module(f'linear_{i+1}', nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1], bias))
            if batch_norm:
                self.layers.add_module(f'bn_{i+1}', nn.BatchNorm1d(hidden_layer_sizes[i+1]))
            self.layers.add_module(f'act_{i+1}', self._get_activation(activation))
            if dropout_prob:
                self.layers.add_module(f'dropout_{i+1}', nn.Dropout(dropout_prob))
        self.layers.add_module(f'linear_{i+1}', nn.Linear(hidden_layer_sizes[-1], output_size, bias))

        if last_activation:
            self.layers.add_module('last_act', self._get_activation(last_activation)) """

    def forward(self, x):
        x = x.flatten(1) if self.flatten_input else x
        return self.layers(x)


class ConvolutionalBlock(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, want_shortcut: bool, downsample: bool, last_layer: bool, pool_type: str, activation: str):
        super().__init__()

        self.want_shortcut = want_shortcut
        self.activation = self._get_activation(activation)
        conv_stride = 1
        pooling = None
        tag = None

        if downsample:
            if last_layer:
                self.want_shortcut = False
                pooling = 'adaptivemaxpool'
                kw = dict(output_size=2)
                hook = True
            else:
                if pool_type == 'convolution':
                    conv_stride = 2     # dimezza la grandezza della feature map
                elif pool_type == 'kmax':
                    channels = [64, 128, 256, 512]
                    dimension = [511, 256, 128]
                    index = channels.index(in_channels)
                    pooling = 'adaptivemaxpool'
                    kw = dict(output_size=dimension[index])
                else:
                    pooling = 'maxpool'
                    kw = dict(kernel_size=3, stride=2, padding=1)
        
        self.block_layers = nn.Sequential(OrderedDict({
            'conv_1': nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=conv_stride, padding=1, bias=False),
            'bn_1': nn.BatchNorm2d(in_channels),
            'act_1': self.activation,
            'conv_2': nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            'bn_2': nn.BatchNorm2d(out_channels),
            'act_2': self.activation
        }))
        if pooling is not None:
            name = 'pool_hook' if hook else 'pool'
            self.block_layers.add_module(name, self._get_pooling(pooling, kw))
        
        if self.want_shortcut:
            self.shortcut = nn.Sequential(OrderedDict({
                'projection': nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                'bn': nn.BatchNorm2d(out_channels)
            }))

        # self.relu = nn.ReLU()

    def forward(self, x):
        if self.want_shortcut:
            short = x
            out = self.block_layers(x)
            if out.shape != short.shape:
                short = self.shortcut(short)
            out = self.activation(short + out)  # self.relu(short + out)
            return out
        else:
            return self.block_layers(x)


class ConvolutionalNeuralNetwork(BaseModel):
    def __init__(self, depth: int, n_classes: int, want_shortcut: bool, pool_type: str, activation: str, fc_activation: str):
        super().__init__()
        channels = [64, 128, 256, 512]
        if depth == 9:
            num_conv_block = [1, 1, 1, 1]   # contains the number of conv blocks for each stage of the network
        elif depth == 17:
            num_conv_block = [2, 2, 2, 2]
        elif depth == 32:
            num_conv_block = [4, 4, 4, 4]
        else:
            num_conv_block = [6, 6, 6, 6]

        self.conv_net = nn.Sequential(OrderedDict({
            'init_conv': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        }))

        layer_counter = 0
        for x in range(len(num_conv_block)):    # Per ogni stadio della rete
            for i in range(num_conv_block[x]):  # Per ogni blocco dello stadio
                layer_counter += 1
                if num_conv_block[x] - 1 == i:  # Se siamo all'ultimo blocco dello stadio
                    if len(num_conv_block) - 1 == x:    # Se siamo all'ultimo stadio (e quindi all'ultimo blocco dell'ultimo stadio)
                        self.conv_net.add_module(f'conv_block_{layer_counter}', ConvolutionalBlock(channels[x], channels[x], want_shortcut, True, True, pool_type, activation))
                    else:
                        self.conv_net.add_module(f'conv_block_{layer_counter}', ConvolutionalBlock(channels[x], channels[x] * 2, want_shortcut, True, False, pool_type, activation))
                else:
                    self.conv_net.add_module(f'conv_block_{layer_counter}', ConvolutionalBlock(channels[x], channels[x], want_shortcut, False, False, pool_type, activation))

        self.fc = MultiLayerPerceptron(
            n_hidden_layers=2, 
            input_size=2048, 
            hidden_layer_sizes=[1024, 1024], 
            output_size=n_classes, 
            activation=fc_activation, 
            batch_norm=True, 
            dropout_prob=0, 
            last_activation=None
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


""" class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_name = None
        self.net = None
    

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
        pass
 """


class MultiLayerPerceptron(nn.Module):
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
        self.last_activation
        self.flatten_input = flatten_input
        
        assert n_hidden_layers == len(hidden_layer_sizes), "Number of hidden layers doesn't match with given hidden sizes."
    
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        for i in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_layer_sizes[i+1]))
            self.layers.append(self._get_activation())
            if dropout_prob:
                self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))

        if last_activation:
            self.layers.append(self._get_activation(last_activation))

    def _get_activation(self):
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation.lower() == 'tanh':
            return nn.Tanh()
        elif self.activation.lower() == 'softmax':
            return nn.Softmax(dim=1)
        else:
            raise NotImplementedError("Activation function '{}' is not implemented.".format(self.activation))

    def forward(self, x):
        x = x.flatten(1) if self.flatten_input else x
        return self.layers(x)


""" class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )

    def forward(self, x):
        return self.layer(x) """


def conv3x3(in_channels:int , out_channels: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual_conn, downsample, last_layer, pool_type):
        super().__init__()

        self.residual_conn = residual_conn
        if self.residual_conn:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)

        if downsample:
            if last_layer:
                self.residual_conn = False
                self.layers.append(nn.AdaptiveMaxPool2d(2))
            else:
                if pool_type == 'convolution':
                    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                           kernel_size=3, stride=2, padding=1, bias=False)
                elif pool_type == 'kmax':
                    channels = [64, 128, 256, 512]
                    dimension = [511, 256, 128]
                    index = channels.index(in_channels)
                    self.layers.append(nn.AdaptiveMaxPool2d(dimension[index]))
                else:
                    self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.residual_conn:
            short = x
            out = self.conv1(x)
            out = self.layers(out)
            if out.shape != short.shape:
                short = self.shortcut(short)
            out = self.relu(short + out)
            return out
        else:
            out = self.conv1(x)
            return self.layers(out)


class FullyConnectedBlock(nn.Module):
    def __init__(self, 
            n_hidden_layers: int, 
            input_size: int, 
            hidden_layer_sizes: List[int], 
            output_size: int, 
            activation: str, 
            batch_norm: bool = False, 
            dropout_prob: float = None,
            last_activation: str = None,
            flatten_input: bool = False
        ):
        super().__init__(
            self, 
            n_hidden_layers, 
            input_size, 
            hidden_layer_sizes, 
            output_size, 
            activation, 
            batch_norm, 
            dropout_prob,
            last_activation,
            flatten_input
        )


class ConvolutionalNeuralNetworks(nn.Module):
    def __init__(self, depth, n_classes, want_shortcut, pool_type):
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

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
        )

        last_layer = False

        for x in range(len(num_conv_block)):    # Per ogni sezione della rete
            for i in range(num_conv_block[x]):  # Per ogni blocco dello stadio
                if num_conv_block[x] - 1 == i:  # Se siamo all'ultimo blocco dello stadio
                    if len(num_conv_block) - 1 == x:    # Se siamo all'ultimo stadio (e quindi all'ultimo blocco dell'ultimo stadio)
                        last_layer = True
                        self.layers.append(
                            ConvolutionalBlock(channels[x], channels[x], want_shortcut, True, last_layer, pool_type))
                    else:
                        self.layers.append(ConvolutionalBlock(channels[x], channels[x] * 2,
                                                                  want_shortcut, True, last_layer, pool_type))
                else:
                    self.layers.append(ConvolutionalBlock(channels[x], channels[x],
                                                              want_shortcut, False, last_layer, pool_type))

        self.fc = FullyConnectedBlock(
            n_hidden_layers=2, 
            input_size=2048, 
            hidden_layer_sizes=[1024,1024], 
            output_size=n_classes, 
            activation='relu', 
            batch_norm=True, 
            dropout_prob=0, 
            last_activation=None
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    



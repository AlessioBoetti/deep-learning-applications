from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
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
        

class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_hidden_layers: int, input_size: int, hidden_layer_sizes: List[int], output_size: int, activation: str, batch_norm: bool = False, dropout_prob: float = None):
        super().__init__()
        
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = True if dropout_prob else False
        self.dropout_prob = dropout_prob
        
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
        return self.layers(x.flatten(1))
    



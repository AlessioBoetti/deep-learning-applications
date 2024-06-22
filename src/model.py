from typing import List, Tuple
import math
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, TaskType, get_peft_model


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
    
    
    @torch.no_grad()
    def _init_weights(self, modules, mode: str = None, dist: str = None, mean: float = 0.0, std: float = 0.02):
        # From https://pytorch.org/docs/stable/notes/modules.html#modules-as-building-blocks
        if not isinstance(modules, str):
            modules = list(modules)
        
        for module in modules:
            if mode in ['xavier', 'glorot']:
                if dist == 'normal':
                    nn.init.xavier_normal_(module.weight)
                elif dist == 'uniform':
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise NotImplementedError
            elif mode in ['kaiming']:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            else:
                if dist == 'normal':
                    torch.nn.init.normal_(module.weight, mean=mean, std=std)
                else:
                    raise NotImplementedError
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # oppure module.bias.fill_(0.0)


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
            raise NotImplementedError("Activation function '{}' is not implemented.".format(activation))
    

    def _get_pooling(self, pooling, kw):
        if pooling.lower() == 'adaptivemaxpool':
            return nn.AdaptiveMaxPool2d(**kw)
        elif pooling.lower() == 'maxpool':
            return nn.MaxPool2d(**kw)
        else:
            raise NotImplementedError("Activation function '{}' is not implemented.".format(pooling))


class MultiLayerPerceptron(BaseModel):
    def __init__(
            self, 
            n_hidden_layers: int, 
            input_size: int, 
            hidden_layer_sizes: List[int], 
            output_size: int, 
            activation: str, 
            batch_norm: bool = False, 
            dropout: float = None,
            last_activation: str = None,
            flatten_input: bool = False,
        ):
        super().__init__()
        
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
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
            if dropout:
                self.layers.add_module(f'dropout_{i+1}', nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers.add_module(f'last_linear', nn.Linear(hidden_layer_sizes[-1], output_size, bias))
        if last_activation:
            self.layers.add_module('last_act', self._get_activation(last_activation))


    def forward(self, x):
        x = x.flatten(1) if self.flatten_input else x
        return self.layers(x)


class ConvolutionalBlock(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, want_shortcut: bool, downsample: bool, last_layer: bool, pool_type: str, activation: str):
        super().__init__()

        self.want_shortcut = want_shortcut
        conv_stride = 1
        hook = None

        if downsample:
            if last_layer:
                self.want_shortcut = False
                pool_type = 'adaptivemaxpool'
                kw = dict(output_size=2)
                hook = True
            else:
                if pool_type == 'convolution':
                    conv_stride = 2     # dimezza la grandezza della feature map
                    pool_type = None
                elif pool_type == 'adaptivemaxpool':
                    channels = [64, 128, 256, 512]
                    dimension = [511, 256, 128]
                    index = channels.index(in_channels)
                    kw = dict(output_size=dimension[index])
                elif pool_type == 'maxpool':
                    kw = dict(kernel_size=3, stride=2, padding=1)
                else:
                    raise NotImplementedError('Valid pool types are "convolution", "adaptivemaxpool" or "maxpool"')
        else:
            pool_type = None

        self.block_layers = nn.Sequential(OrderedDict({
            'conv_1': nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=conv_stride, padding=1, bias=False),
            'bn_1': nn.BatchNorm2d(in_channels),
            'act_1': self._get_activation(activation),
            'conv_2': nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            'bn_2': nn.BatchNorm2d(out_channels),
            'act_2': self._get_activation(activation)
        }))
        if pool_type:
            name = 'pool_hook' if hook else 'pool'
            self.block_layers.add_module(name, self._get_pooling(pool_type, kw))
        
        if self.want_shortcut:
            self.shortcut = nn.Sequential(OrderedDict({
                'projection': nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                'bn': nn.BatchNorm2d(out_channels)
            }))
            self.activation = self._get_activation(activation)
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
    def __init__(self, depth: int, output_size: int, want_shortcut: bool, pool_type: str, activation: str, fc_activation: str):
        super().__init__()
        channels = [64, 128, 256, 512]
        if depth == 9:  # 9 because there are 4 ConvBlocks, each composed of 2 Conv layers, + the init_conv layer
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
            output_size=output_size, 
            activation=fc_activation, 
            batch_norm=True, 
            dropout=0, 
            last_activation=None
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BigramLanguageModel(nn.Module):
    # From https://www.youtube.com/watch?v=kCc8FmEb1nY
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Head(nn.Module):
    """ One head of Self-Attention """

    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # Input of size (batch, time-step, channels)
        # Output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs) (hs is head size)
        q = self.query(x)  # (B, T, hs)
        # Compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of Self-Attention in parallel """

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)
    

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, block_size, dropout):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.ln1 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size, dropout)  # sa sta per self-attention
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedFoward(n_embed, dropout)
        

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(BaseModel):
    # From https://www.youtube.com/watch?v=kCc8FmEb1nY
    def __init__(self, vocab_size, n_embed, block_size, n_head, n_layers, dropout):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head, block_size, dropout) for _ in range(n_layers)])
        self.layer_norm_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)


    def forward(self, idx, device):
        B, T = idx.shape  # idx is a (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb                   # (B, T, C)
        x = self.blocks(x)                      # (B, T, C)
        x = self.layer_norm_final(x)            # (B, T, C)
        logits = self.lm_head(x)                # (B, T, vocab_size)
        return logits


    def generate(self, idx, max_new_tokens, block_size, device):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits = self(idx_cond, device)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class BERT(BaseModel):
    def __init__(self, model_name, output_size, init_mode, init_dist, cache_dir, torch_dtype=torch.bfloat16, device_map='auto', freeze_model_base: bool = False, peft: str = None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, return_dict=True, cache_dir=cache_dir, device_map=device_map)  # torch_dtype=torch_dtype
        self.hidden = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, output_size)
        self._init_weights([self.hidden, self.classifier], init_mode, init_dist)

        if freeze_model_base:
            self.model.requires_grad_(False)
            
        if peft:
            # From https://huggingface.co/docs/peft/quicktour
            if peft['method'] == 'LoRA':
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, 
                    inference_mode=False, 
                    r=peft['r'], 
                    lora_alpha=peft['alpha'], 
                    lora_dropout=peft['dropout'],
                    use_rslora=peft['rslora']
                )
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()

            # elif peft == 'ReFT'
                # # From https://www.youtube.com/watch?v=iy9Z4DyHxvE
                # reft_config = pyreft.ReftConfig(
                #     representations={
                #         'layer':4,
                #         'component':'block_output', 
                #         'low_rank_dimension':4,
                #         'intervention':pyreft.LoreftIntervention(
                #             embed_dim=model.config.hidden_size, low_rank_dimension=4
                #         ) 
                #     }
                # )
                # reft_model = pyreft.get_reft_model(model, reft_config)
                # reft_model.set_device(device)
                # data_module = pyreft.make_last_position_supervised_data_module(
                #     tokenizer, 
                #     model, 
                #     [prompt_template(x) for x in X], 
                #     y 
                # ) 
            else:
                raise NotImplementedError()


    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = torch.mean(output.last_hidden_state, 1)
        output = F.relu(self.hidden(output))
        output = self.classifier(output)
        output = F.softmax(output)
        return output


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERT('distilroberta-base', output_size=10, init_mode='xavier', dist='normal', cache_dir='./hf').to(device)
    input_ids = torch.randint(0, 1000, (32, 128)).to(device)
    attention_mask = torch.randint(0, 2, (32, 128)).to(device)
    print(model(input_ids, attention_mask).shape)
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from adversarial import *


class MaxLogitsPostprocessor():
    def __init__(self):
        pass


    @torch.no_grad()
    def postprocess(self, model, inputs, *args):
        outputs = model(inputs)
        conf, pred = torch.max(outputs, dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()


class MaxSoftmaxPostprocessor():
    """ Maximum Softmax Probability (MSP) """
    def __init__(self):
        pass


    @torch.no_grad()
    def postprocess(self, model, inputs, *args):
        outputs = model(inputs)
        conf, pred = F.softmax(outputs, dim=1).max(dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()


class ODINPostprocessor():
    def __init__(self, temperature: float = 1000.0, noise: float = 0.0014):
        if temperature == 0.0:
            temperature += 1e-7
        if noise == 0.0:
            noise += 0.001
        self.temperature = temperature
        self.noise = noise


    def postprocess(self, model, inputs, labels, criterion, scaler, adv_cfg):
        # inputs.requires_grad = True
        # outputs = model(inputs)

        # # Calculating the perturbation we need to add, that is,
        # # the sign of gradient of cross entropy loss w.r.t. input
        # targets = outputs.argmax(axis=1)

        # # Using temperature scaling
        # outputs = outputs / self.temperature

        # loss = criterion(outputs, targets)
        # scaler.scale(loss).backward()

        # # Normalizing the gradient to binary in {0, 1}
        # gradient = torch.ge(inputs.grad.detach(), 0)  # torch.ge: greater or equal
        # gradient = (gradient.float() - 0.5) * 2

        # # Scaling values taken from original code
        # # gradient = gradient/std

        # # Adding small perturbations to images
        # adv_inputs = torch.add(inputs.detach(), gradient, alpha=-self.noise)  # torch.add(input, other, alpha): adds other, scaled by alpha, to input.
        
        adv_cfg.update({'temperature': self.temperature, 'alpha': self.noise, 'fast': False})
        delta = attack(model, inputs, labels, scaler=scaler, **adv_cfg)
        adv_inputs = inputs + delta

        with torch.no_grad():
            outputs = model(adv_inputs)
        outputs = outputs / self.temperature

        # Calculating the confidence after adding perturbations
        outputs = outputs.detach()
        outputs = outputs - outputs.max(dim=1, keepdims=True).values
        outputs = outputs.exp() / outputs.exp().sum(dim=1, keepdims=True)

        conf, pred = outputs.max(dim=1)
        model.zero_grad()

        return pred.cpu().numpy(), conf.cpu().numpy()


    def set_hyperparam(self, temperature, noise):
        self.temperature = temperature
        self.noise = noise


    def get_hyperparam(self):
        return self.temperature, self.noise


class CEA():
    # From https://github.com/mazizmalayeri/CEA
    
    def __init__(self, model, processor, loader, criterion, scaler, adv_cfg, device, percentile_top, addition_coef, threshold_caution_coef=1.1, hook_name: str = 'penultimate'):
        """
            Args:
                device (str): Device for computation.
                processor (callable): Function for calculating original novelty score, e.g., MSP.
                percentile_top (float): p parameter in CEA used for calculating τ.
                addition_coef (float): γ parameter in CEA used for calculating λ.
                threshold_caution_coef (float, optional): ρ parameter in CEA used for calculating λ.
        """
    
        self.processor = processor
        self.percentile_top = percentile_top  # p in CEA
        self.threshold_caution_coef = threshold_caution_coef  # ρ in CEA
        self.addition_coef = addition_coef  # γ in  CEA
        self.coef = None  # λ in CEA
        self.threshold_top = None  # τ in CEA
        self.setup_done = False
        self.setup(model, loader, criterion, scaler, adv_cfg, device, hook_name)


    def setup(self, model, loader, criterion, scaler, adv_cfg, device, hook_name):
        """
            Calculating hyperparametrs based on a validation set from ID.
        """
        if not self.setup_done:
            activations_list = []
            original_scores = []
            added_scores = []
            model.eval()

            # activation = {}
            # def get_activation(name):
            #     def hook(model, input, output):
            #         activation[name] = output.detach()
            #     return hook
            
            # model.fc.layers.act_2.register_forward_hook(get_activation(hook_name))

            for batch in loader:
                inputs, labels, idx = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # outputs = model(inputs)
                # activations_list.append(activation[hook_name])
                # activation[hook_name] = None
                
                with torch.no_grad():
                    x = model.conv_net(inputs)
                    x = x.view(x.size(0), -1)
                    x = model.fc.layers.act_1(model.fc.layers.bn_1(model.fc.layers.linear_1(x)))
                    act = model.fc.layers.act_2(model.fc.layers.bn_2(model.fc.layers.linear_2(x)))
                activations_list.append(act.detach().cpu())

                _, original_score = self.processor.postprocess(model, inputs, labels, criterion, scaler, adv_cfg)
                original_scores.append(original_score)

            self.activations_list = np.concatenate(activations_list, axis=0)
            self.set_threshold()

            for activation in activations_list:
                nov = activation - self.threshold_top
                nov = nov.clip(min=0)
                nov = -torch.norm(nov, dim=1)
                added_scores.append(nov)  # novelty score
            
            self.added_scores = np.concatenate(added_scores, axis=0)
            self.original_scores = np.concatenate(original_scores, axis=0)
            self.set_coef()
            
            self.setup_done = True


    def postprocess(self, model, inputs, labels, criterion, scaler, adv_cfg):
        """
            Calculating the novelty score on data.
        """
        
        # Compute original novelty score
        pred, conf = self.processor.postprocess(model, inputs, labels, criterion, scaler, adv_cfg)
        
        # Compute CEA added value
        # outputs = model(inputs)
        # act = activation[hook_name]

        with torch.no_grad():
            x = model.conv_net(inputs)
            x = x.view(x.size(0), -1)
            x = model.fc.layers.act_1(model.fc.layers.bn_1(model.fc.layers.linear_1(x)))
            act = model.fc.layers.act_2(model.fc.layers.bn_2(model.fc.layers.linear_2(x)))
        
        nov = act - self.threshold_top
        nov = nov.clip(min=0)
        nov = -torch.norm(nov, dim=1)
        
        return pred, self.coef * nov.cpu().numpy() + conf


    def set_threshold(self):
        """
            Set threshold τ for capturing extreme activations.
        """
        
        self.threshold_top = self.threshold_caution_coef * np.percentile(self.activations_list.flatten(), self.percentile_top)
        # print('Top threshold at percentile {} over ID data is: {}'.format(self.percentile_top, self.threshold_top))


    def set_coef(self): 
        """
            Set coefficient λ for adding CEA to original novelty score.
        """
  
        # print(self.original_scores.mean(), self.added_scores.mean())
        self.coef = self.addition_coef * abs(self.original_scores.mean()) / (abs(self.added_scores.mean())+1e-1)
        # print('Coeffient of added novelty score is: {}'.format(self.coef))


def get_ood_scores(model, id_loader, ood_loader, score_function, criterion, scaler, adv_cfg, device, hook_name: str = 'penultimate', missclass_as_ood: bool = False):
    """
        Calculate the novelty scores that an OOD detector (score_function) assigns to ID and OOD and evaluate them via AUROC and FPR.

        Parameters:
        -----------
        model: The neural network model for applying the post-hoc method.
        id_loader
        ood_loader
        score_function: The scoring function that assigns each sample a novelty score.
        hook_name
        device: The device on which to run the model (e.g., 'cpu' or 'cuda').
        missclass_as_ood: If True, consider misclassified in-distribution samples as OOD. Default is False.
    """
    
    model.eval()
    torch.cuda.empty_cache()
    
    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook

    # model.fc.layers.act_2.register_forward_hook(get_activation(hook_name))

    y_pred_id, scores_id, y_true_id = [], [], []
    for batch in id_loader:
        inputs, labels, idx = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred, conf = score_function(model, inputs, labels, criterion, scaler, adv_cfg)
        y_pred_id += list(pred)
        scores_id += list(conf)
        y_true_id += list(labels.cpu().detach().numpy())
    
    torch.cuda.empty_cache()

    y_pred_ood, scores_ood, y_true_ood = [], [], []
    for batch in ood_loader:
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred, conf = score_function(model, inputs, labels, criterion, scaler, adv_cfg)
            y_pred_ood += list(pred)
            scores_ood += list(conf)
            y_true_ood += list(np.ones(conf.shape[0])*-1)

    torch.cuda.empty_cache()

    return (y_true_id, y_pred_id, scores_id), (y_true_ood, y_pred_ood, scores_ood)

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
import torch


class MaxLogitPostprocessor():
    def __init__(self):
        pass


    @torch.no_grad()
    def postprocess(self, model, inputs):
        output = model(inputs)
        conf, pred = torch.max(output, dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()



class ODINPostprocessor():
    def __init__(self, temperature: float = 1000.0, noise: float = 0.0014):
        self.temperature = temperature
        self.noise = noise


    def postprocess(self, model, inputs, criterion):
        inputs.requires_grad = True
        outputs = model(inputs)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        targets = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(outputs, targets)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.detach(), 0)  # torch.ge: greater or equal
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        # gradient = gradient/std

        # Adding small perturbations to images
        temp_inputs = torch.add(inputs.detach(), gradient, alpha=-self.noise)  # torch.add(input, other, alpha): adds other, scaled by alpha, to input.
        outputs = model(temp_inputs)
        outputs = outputs / self.temperature

        # Calculating the confidence after adding perturbations
        outputs = outputs.detach()
        outputs = outputs - outputs.max(dim=1, keepdims=True).values
        outputs = outputs.exp() / outputs.exp().sum(dim=1, keepdims=True)

        conf, pred = outputs.max(dim=1)

        return pred.cpu().numpy(), conf.cpu().numpy()


    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]


    def get_hyperparam(self):
        return [self.temperature, self.noise]



class CEA():
    # From https://github.com/mazizmalayeri/CEA
    
    def __init__(self, model, processor, loader, device, percentile_top, addition_coef, treshold_caution_coef=1.1, hook_name: str = 'penultimate'):
        """
            Args:
                device (str): Device for computation.
                processor (callable): Function for calculating original novelty score, e.g., MSP.
                percentile_top (float): p parameter in CEA used for calculating τ.
                addition_coef (float): γ parameter in CEA used for calculating λ.
                treshold_caution_coef (float, optional): ρ parameter in CEA used for calculating λ.
        """
    
        self.processor = processor
        self.percentile_top = percentile_top  # p in CEA
        self.treshold_caution_coef = treshold_caution_coef  # ρ in CEA
        self.addition_coef = addition_coef  # γ in  CEA
        self.coef = None  # λ in CEA
        self.threshold_top = None  # τ in CEA
        self.setup(model, loader, device, hook_name)


    def setup(self, model, loader, device, hook_name):
        """
            Calculating hyperparametrs based on a validation set from ID.
        """
        
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

        with torch.no_grad():
            for batch in loader:
                inputs, labels, idx = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # outputs = model(inputs)
                # activations_list.append(activation[hook_name])
                # activation[hook_name] = None

                x = model.conv_net(inputs)
                x = x.view(x.size(0), -1)
                x = model.fc.layers.act_1(model.fc.layers.bn_1(model.fc.layers.linear_1(x)))
                act = model.fc.layers.act_2(model.fc.layers.bn_2(model.fc.layers.linear_2(x)))
                activations_list.append(act)

                _, original_score = self.processor.postprocess(model, inputs)
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


    @torch.no_grad()
    def postprocess(self, model, inputs):
        """
            Calculating the novelty score on data.
        """
        
        # Compute original novelty score
        pred, conf = self.processor.postprocess(model, inputs)
        
        # Compute CEA added value
        # outputs = model(inputs)
        # act = activation[hook_name]

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
        
        self.threshold_top = self.treshold_caution_coef * np.percentile(self.activations_list.flatten(), self.percentile_top)
        # print('Top threshold at percentile {} over ID data is: {}'.format(self.percentile_top, self.threshold_top))


    def set_coef(self): 
        """
            Set coefficient λ for adding CEA to original novelty score.
        """
  
        # print(self.original_scores.mean(), self.added_scores.mean())
        self.coef = self.addition_coef * abs(self.original_scores.mean()) / (abs(self.added_scores.mean())+1e-1)
        # print('Coeffient of added novelty score is: {}'.format(self.coef))


def acc(y_pred, y_true):
    ind_pred = y_pred[y_true != -1]
    ind_label = y_true[y_true != -1]
    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)
    return acc


def auc_and_fpr_recall(y_true, max_scores, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(y_true)
    ood_indicator[y_true == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = roc_curve(ood_indicator, -max_scores)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in = precision_recall_curve(1 - ood_indicator, max_scores)
    precision_out, recall_out, thresholds_out = precision_recall_curve(ood_indicator, -max_scores)

    auroc = auc(fpr_list, tpr_list)
    aupr_in = auc(recall_in, precision_in)
    aupr_out = auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


def compute_all_metrics(y_true, y_pred, max_scores):
    # np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(y_true, max_scores, recall)
    # accuracy = acc(y_pred, y_true)
    accuracy = accuracy_score(y_true, y_pred)
    return {'FPR at 0.95 TPR': fpr, 'auc_roc': auroc, 'average_precision_id': aupr_in, 'average_precision_ood': aupr_out, 'accuracy': accuracy}


def compute_ood_metrics(pred_id, conf_id, gt_id, pred_ood, conf_ood, gt_ood, missclass_as_ood=False):
    """
        Calculates the OOD metrics (fpr, auroc, etc.) based on the postprocessing results.
        
        Parameters:
        -----------
        missclass_as_ood: bool, optional
            If True, consider misclassified in-distribution samples as OOD. Default is False.
    
        Returns:
        --------
        dict:
            A dictionary containing various OOD detection evaluation metrics.
    """
  
    if missclass_as_ood:
        id_gt_np = np.array(gt_id)
        id_gt_np[np.array(pred_id) != id_gt_np] = -1
        # print((id_gt_np == -1).mean())
        gt_id = id_gt_np.tolist()
    
    y_true = np.concatenate([gt_id, gt_ood])
    y_pred = np.concatenate([pred_id, pred_ood])
    max_scores = np.concatenate([conf_id, conf_ood])
    
    check_nan = np.isnan(max_scores)
    check_inf = np.isinf(max_scores)
    for check in [check_nan, check_inf]:
        num_check = check.sum()
        if num_check > 0:
            y_true = np.delete(y_true, np.where(check))
            y_pred = np.delete(y_pred, np.where(check))
            max_scores = np.delete(max_scores, np.where(check))

    ood_metrics = compute_all_metrics(y_true, y_pred, max_scores)
    return ood_metrics


def get_ood_score(model, id_loader, ood_loader, score_function, device, hook_name: str = 'penultimate', missclass_as_ood: bool = False):
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
    
    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook

    # model.fc.layers.act_2.register_forward_hook(get_activation(hook_name))

    preds_id, confs_id, gt_id = [], [], []
    for batch in id_loader:
        inputs, labels, idx = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred, conf = score_function(model, inputs)
        preds_id += list(pred)
        confs_id += list(conf)
        gt_id += list(labels.cpu().detach().numpy())

    preds_ood, confs_ood, gt_ood = [], [], []
    for batch in ood_loader:
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred, conf = score_function(model, inputs)
            preds_ood += list(pred)
            confs_ood += list(conf)
            gt_ood += list(np.ones(conf.shape[0])*-1)
    
    metrics = compute_ood_metrics(preds_id, confs_id, gt_id, preds_ood, confs_ood, gt_ood, missclass_as_ood)
    # auc = ood_score_calc(scores_inlier, scores_ood)

    return metrics

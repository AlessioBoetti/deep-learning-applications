import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch


class MaxLogitPostprocessor():
    def __init__(self):
        pass

    @torch.no_grad()
    def postprocess(self, model, inputs):
        output = model(inputs)
        conf, pred = torch.max(output, dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()


class CEA():
    # From https://github.com/mazizmalayeri/CEA
    
    def __init__(self, model, processor, loader, device, percentile_top, addition_coef, hook_name, treshold_caution_coef=1.1):
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

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        model.penultimate_layer.register_forward_hook(get_activation(hook_name))

        with torch.no_grad():
            for batch in loader:
                inputs, labels, idx = batch
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                activations_list.append(activation[hook_name])
                activation[hook_name] = None

                _, original_score = self.processor(model, inputs)
                original_scores.append(original_score)

        self.activations_list = np.concatenate(activations_list, axis=0)
        self.set_threshold()

        added_scores = []
        for activation in activations_list:
            nov = activation - self.threshold_top
            nov = nov.clip(min=0)
            nov = -torch.norm(nov, dim=1)
            added_scores.append(nov)  # novelty score
        
        self.added_scores = np.concatenate(added_scores, axis=0)
        self.original_scores = np.concatenate(original_scores, axis=0)
        self.set_coef()
        
        # with torch.no_grad():
        #     for batch in loader:
        #         inputs, labels, idx = batch
        #         inputs = inputs.to(device, non_blocking=True)
        #         labels = labels.to(device, non_blocking=True)
                
        #         added_score, original_score = self.postprocess(model, inputs, val=True)
        #         added_scores.append(added_score)
        #         original_scores.append(original_score)

        # self.added_scores = np.concatenate(added_scores, axis=0)
        # self.original_scores = np.concatenate(original_scores, axis=0)
        # self.set_coef()


    @torch.no_grad()
    def postprocess(self, model, inputs, activation, hook_name):
        """
            Calculating the novelty score on data.
        """
        
        # Compute original novelty score
        pred, conf = self.processor(model, inputs)
        
        # Compute CEA added value
        outputs = model(inputs)
        act = activation[hook_name]
        nov = act - self.threshold_top
        nov = nov.clip(min=0)
        nov = -torch.norm(nov, dim=1)
        
        return pred, self.coef * nov.cpu().numpy() + conf, activation


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


def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]
    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)
    return acc


def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in = precision_recall_curve(1 - ood_indicator, conf)
    precision_out, recall_out, thresholds_out = precision_recall_curve(ood_indicator, -conf)

    auroc = auc(fpr_list, tpr_list)
    aupr_in = auc(recall_in, precision_in)
    aupr_out = auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


def compute_all_metrics(conf, label, pred):
    # np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)
    accuracy = acc(pred, label)
    return fpr, auroc, aupr_in, aupr_out, accuracy


def eval_ood(pred_id, conf_id, gt_id, pred_ood, conf_ood, gt_ood, missclass_as_ood=False):
    """
        Calculates the OOD metrics (fpr, auroc, etc.) based on the postprocessing results.
        
        Parameters:
        -----------
        postprocess_results: list
            A list containing the following elements in order:
            [id_pred, id_conf, ood_pred, ood_conf, id_gt, ood_gt].
        to_print: bool, optional
            Whether to print the evaluation metrics or only return the metrics. Default is True.
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
        
    pred = np.concatenate([pred_id, pred_ood])
    conf = np.concatenate([conf_id, conf_ood])
    label = np.concatenate([gt_id, gt_ood])
    
    check_nan = np.isnan(conf)
    check_inf = np.isinf(conf)
    for check in [check_nan, check_inf]:
        num_check = check.sum()
        if num_check > 0:
            conf = np.delete(conf, np.where(check))
            pred = np.delete(pred, np.where(check))
            label = np.delete(label, np.where(check))

    ood_metrics = compute_all_metrics(conf, label, pred)
    return ood_metrics


def get_ood_score(model, id_loader, ood_loader, score_function, hook_name, device, missclass_as_ood=False):
    """
        Calculate the novelty scores that an OOD detector (score_function) assigns to ID and OOD and evaluate them via AUROC and FPR.

        Parameters:
        -----------
        model: torch.nn.Module or None
            The neural network model for applying the post-hoc method.
        in_test_features: torch.Tensor
            In-distribution test features.
        in_test_labels: torch.Tensor
            In-distribution test labels.
        ood_type: str
            The type of out-of-distribution (OOD) data ('other_domain', 'feature_separation', or 'multiplication').
        score_function: callable
            The scoring function that assigns each sample a novelty score.
        batch_size: int
            Batch size for processing data.
        device: str
            The device on which to run the model (e.g., 'cpu' or 'cuda').
        preprocess: object
            The preprocess for normalizing the data if it is needed.
        random_sample: list or None, optional
            List of randomly selected feature indices for 'multiplication'. Default is None.
        scales: list or None, optional
            List of scales for feature multiplication. Default is None.
        out_features: torch.Tensor or None, optional
            Out-of-distribution (OOD) features for 'other_domain' or 'feature_separation'. Default is None.
        missclass_as_ood: bool, optional
            If True, consider misclassified in-distribution samples as OOD. Default is False.
    """
    
    model.eval() 
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.penultimate_layer.register_forward_hook(get_activation(hook_name))

    preds_id, confs_id, gt_id = [], [], []
    for batch in id_loader:
        inputs, labels, idx = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred, conf, activation = score_function(model, inputs, activation, hook_name)
        activation[hook_name] = None
        preds_id += list(pred)
        confs_id += list(conf)
        gt_id += list(labels.cpu().detach().numpy())

    
    preds_ood, confs_ood, gt_ood = [], [], []
    for batch in ood_loader:
            inputs, labels, idx = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred, conf = score_function(model, inputs, activation, hook_name)
            activation[hook_name] = None
            preds_ood += list(pred)
            confs_ood += list(conf)
            gt_ood += list(np.ones(conf.shape[0])*-1)
    
    eval_ood(preds_id, confs_id, gt_id, preds_ood, confs_ood, gt_ood, missclass_as_ood)
    # auc = ood_score_calc(scores_inlier, scores_ood)

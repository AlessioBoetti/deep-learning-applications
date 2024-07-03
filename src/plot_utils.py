import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn.functional as F

from utils import *


def plot_confusion_matrix(y_true, y_pred, classes, out_path, split):
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(np.float32)
    cmn /= cmn.sum(1)
    cmn = (100*cmn).astype(np.int32)
    disp = ConfusionMatrixDisplay(cmn, display_labels=classes)
    disp.plot()
    plt.savefig(f'{out_path}/confusion_matrix_{split}.png')
    plt.close()

    # multilabel_confusion_matrix(y_true, y_pred, labels=classes)


def plot_scores(scores_id, scores_ood, out_path, split, scores_type, T=1):
    suptitle = 'ID vs OOD scores'
    title = f'({scores_type}, T = {T})' if 'softmax' in scores_type else f'({scores_type})'
    filename = f'{out_path}/{split}_{scores_type}_scores'

    plt.figure()
    plt.hist(scores_id, density=True, alpha=0.5, bins=25, label='ID')
    plt.hist(scores_ood, density=True, alpha=0.5, bins=25, label='OOD')
    plt.suptitle(suptitle)
    plt.title(title)
    plt.savefig(f'{filename}_histogram.png')
    plt.close()

    plt.figure()
    plt.plot(sorted(scores_id), label='ID')
    plt.plot(sorted(scores_ood), label='OOD')
    plt.suptitle(suptitle)
    plt.title(title)
    plt.savefig(f'{filename}_plot.png')
    plt.close()


def plot_curves(fpr, tpr, auc_roc, prec_id, rec_id, ap_id, prec_ood, rec_ood, ap_ood, fpr_at_tpr, out_path, split, scores_type):

    filename = f'{out_path}/{split}_{scores_type}'

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.suptitle('ROC curve')
    plt.title('AUC ROC = {0:0.4f} - FPR at 95% TPR = {0:0.4f}'.format(auc_roc, fpr_at_tpr))
    plt.savefig(f'{filename}_roc_curve.png')
    plt.close()

    for recall, precision, ap, dist in zip([rec_id, rec_ood], [prec_id, prec_ood], [ap_id, ap_ood], ['id', 'ood']):
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.ylim([0, 1.05])
        plt.xlim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.suptitle(f'Precision-Recall curve ({dist.upper()} data as positive label)')
        plt.title('AP = {0:0.4f}'.format(ap))
        plt.savefig(f'{filename}_precision_recall_curve_{dist}.png')
        plt.close()


def compute_metrics(y_true, scores, tpr_threshold: float = 0.95):

    y_true = np.array(y_true)
    scores = np.array(scores)

    # accuracy = accuracy_score(y_true, scores)

    # we assume ID samples will have larger score values than OOD samples
    # therefore here we need to negate the scores so that higher scores indicate OOD
    fpr, tpr, _ = roc_curve(y_true, -scores)
    auc_roc = roc_auc_score(y_true, -scores)
    # auc_roc = auc(fpr, tpr)
    fpr_at_tpr = fpr[np.argmax(tpr >= tpr_threshold)]

    # precision and recall curve when ID data have positive label
    precision_id, recall_id, _ = precision_recall_curve(1 - y_true, scores)
    ap_id = average_precision_score(1 - y_true, scores)
    # ap_id = auc(recall_id, precision_id)
    
    # precision and recall curve when OOD data have positive label
    precision_ood, recall_ood, _ = precision_recall_curve(y_true, -scores)
    ap_ood = average_precision_score(y_true, -scores)
    # ap_ood = auc(recall_ood, precision_ood)

    return fpr, tpr, auc_roc, precision_id, recall_id, ap_id, precision_ood, recall_ood, ap_ood, fpr_at_tpr


def extract_results(idx_label_scores, T):
    y_true, y_pred, logit_scores, max_logit_scores, softmax_scores, max_softmax_scores = [], [], [], [], [], []
    for idx, label, scores in idx_label_scores:
        y_true.append(label)
        y_pred.append(np.argmax(scores))
        logit_scores.append(scores)
        max_logit_scores.append(np.max(scores))

        scores_t = torch.tensor(scores, dtype=torch.float)
        soft_scores = F.softmax(scores_t / T, dim=0).numpy().tolist()
        softmax_scores.append(soft_scores)
        max_softmax_scores.append(np.max(soft_scores))
    
    return y_true, y_pred, logit_scores, max_logit_scores, softmax_scores, max_softmax_scores


def get_y_true(y_true_id, y_pred_id, scores_id, scores_ood, missclass_as_ood):
    if missclass_as_ood:
        y_true_id_np = np.array(y_true_id)
        y_true_id_np[np.array(y_pred_id) != y_true_id_np] = 1
        y_true_id_np[np.array(y_pred_id) == y_true_id_np] = 0
        y_true_id = y_true_id_np.tolist()
        y_true = y_true_id
    else:
        y_true = [0] * len(scores_id)
    y_true.extend([1] * len(scores_ood))
    return y_true


def check_values(y_true, y_pred, max_scores):
    check_nan = np.isnan(max_scores)
    check_inf = np.isinf(max_scores)
    for check in [check_nan, check_inf]:
        num_check = check.sum()
        if num_check > 0:
            y_true = np.delete(y_true, np.where(check))
            y_pred = np.delete(y_pred, np.where(check))
            max_scores = np.delete(max_scores, np.where(check))
    return y_true, y_pred, max_scores


def plot_results(idx_label_scores, out_path: str, split: str = None, classes = None, ood_idx_label_scores=None, T: float = 1.0, metrics: dict = None, missclass_as_ood: bool = False, postprocess: str = None, eps=None):
    
    if not postprocess:
        y_true_id, y_pred_id, logit_scores_id, max_logit_scores_id, softmax_scores_id, max_softmax_scores_id = extract_results(idx_label_scores, T)

    if eps:
        img_path = f'{out_path}/imgs_adv/eps_{eps}'
    else:
        img_path = f'{out_path}/imgs'
    create_dirs_if_not_exist(img_path)

    if ood_idx_label_scores is None:
        plot_confusion_matrix(y_true_id, y_pred_id, classes, img_path, split)

        if metrics:
            scores = metrics.pop('scores')
            metrics.update({
                'f1': fbeta_score(y_true_id, y_pred_id, beta=1, average='macro'),
                'softmax_average_precision': average_precision_score(y_true_id, softmax_scores_id, average='macro'),
                'softmax_auc_roc_ovr': roc_auc_score(y_true_id, softmax_scores_id, multi_class='ovr'),
                'softmax_auc_roc_ovo': roc_auc_score(y_true_id, softmax_scores_id, multi_class='ovo'),
            })
            metrics['scores'] = scores
        else:
            metrics = {
                'accuracy': accuracy_score(y_true_id, y_pred_id),
                'precision': precision_score(y_true_id, y_pred_id, average='macro'),
                'recall': recall_score(y_true_id, y_pred_id, average='macro'),
                'f1': fbeta_score(y_true_id, y_pred_id, beta=1, average='macro'),
                'softmax_average_precision': average_precision_score(y_true_id, softmax_scores_id, average='macro'),
                'softmax_auc_roc_ovr': roc_auc_score(y_true_id, softmax_scores_id, multi_class='ovr'),
                'softmax_auc_roc_ovo': roc_auc_score(y_true_id, softmax_scores_id, multi_class='ovo'),
            }
        save_results(out_path, metrics, split, suffix='metrics')

    else:
        if not postprocess:
            y_true_ood, y_pred_ood, logit_scores_ood, max_logit_scores_ood, softmax_scores_ood, max_softmax_scores_ood = extract_results(ood_idx_label_scores, T)
            
            y_true = get_y_true(y_true_id, y_pred_id, max_logit_scores_id, max_logit_scores_ood, missclass_as_ood)
            y_pred = np.concatenate([y_pred_id, y_pred_ood])
            max_logit_scores = max_logit_scores_id + max_logit_scores_ood
            max_softmax_scores = max_softmax_scores_id + max_softmax_scores_ood

            y_true, y_pred, max_logit_scores = check_values(y_true, y_pred, max_logit_scores)
            y_true, y_pred, max_softmax_scores = check_values(y_true, y_pred, max_softmax_scores)

            plot_scores(max_logit_scores_id, max_logit_scores_ood, img_path, split, 'max_logit')
            plot_scores(max_softmax_scores_id, max_softmax_scores_ood, img_path, split, 'max_softmax')

            logit_metrics = compute_metrics(y_true, max_logit_scores)
            plot_curves(*logit_metrics, out_path=img_path, split=split, scores_type='max_logit')

            softmax_metrics = compute_metrics(y_true, max_softmax_scores)
            plot_curves(*softmax_metrics, out_path=img_path, split=split, scores_type='max_softmax')
        else:
            y_true_id, y_pred_id, max_scores_id = idx_label_scores
            y_true_ood, y_pred_ood, max_scores_ood = ood_idx_label_scores
            
            y_true = get_y_true(y_true_id, y_pred_id, max_scores_id, max_scores_ood, missclass_as_ood)
            y_pred = np.concatenate([y_pred_id, y_pred_ood])
            max_scores = max_scores_id + max_scores_ood

            y_true, y_pred, max_scores = check_values(y_true, y_pred, max_scores)

            plot_scores(max_scores_id, max_scores_ood, img_path, split, postprocess)
            metrics = compute_metrics(y_true, max_scores)
            plot_curves(*metrics, out_path=img_path, split=split, scores_type=postprocess)

        # accuracy = accuracy_score(y_true, y_pred)


def save_plot(train_l, train_a, test_l, test_a):
    plt.plot(train_a, '-')
    plt.plot(test_a, '-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid accuracy')
    plt.savefig('result/accuracy')
    plt.close()

    plt.plot(train_l, '-')
    plt.plot(test_l, '-')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig('result/losses')
    plt.close()
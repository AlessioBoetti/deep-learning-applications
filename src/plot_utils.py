import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn.functional as F

from utils import *


def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(np.float32)
    cmn /= cmn.sum(1)
    cmn = (100*cmn).astype(np.int32)
    disp = ConfusionMatrixDisplay(cmn, display_labels=classes)
    disp.plot()
    plt.savefig(f'{out_path}/confusion_matrix.png')
    plt.close()

    # multilabel_confusion_matrix(y_true, y_pred, labels=classes)


def plot_scores(scores_id, scores_ood, out_path, scores_type, T=1):
    plt.figure()
    plt.hist(scores_id, density=True, alpha=0.5, bins=25, label='ID')
    plt.hist(scores_ood, density=True, alpha=0.5, bins=25, label='OOD')
    plt.suptitle(f'ID vs OOD scores')
    title = f'({scores_type})' if scores_type == 'logits' else f'({scores_type}, T = {T})'
    plt.title(title)
    plt.savefig(f'{out_path}/max_{scores_type}_scores_histogram.png')
    plt.close()

    plt.figure()
    plt.plot(sorted(scores_id), label='ID')
    plt.plot(sorted(scores_ood), label='OOD')
    plt.suptitle(f'ID vs OOD scores')
    title = f'({scores_type})' if scores_type == 'logits' else f'({scores_type}, T = {T})'
    plt.title(title)
    plt.savefig(f'{out_path}/max_{scores_type}_scores_plot.png')
    plt.close()


def plot_curves(y_true, scores, out_path, scores_type):

    # ROC Curve & AUC ROC score
    fpr, tpr, _ = roc_curve(y_true, scores)
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    auc_roc = roc_auc_score(y_true, scores)

    # DET Curve
    # fpr, fnr = det_curve(y_true, scores)
    # det_display = DetCurveDisplay(fpr=fpr, fnr=fnr)

    # Precision-Recall Curve & Average Precision score
    prec, recall, _ = precision_recall_curve(y_true, scores)
    # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
    ap = average_precision_score(y_true, scores)

    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.figure()
    plt.step(recall, prec, color='b', alpha=0.2, where='post')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.suptitle('Precision-Recall curve')
    plt.title('AP = {0:0.4f}'.format(ap))
    plt.savefig(f'{out_path}/max_{scores_type}_precision_recall_curve.png')
    plt.close()

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.suptitle('ROC curve')
    plt.title('AUC ROC = {0:0.4f}'.format(auc_roc))
    plt.savefig(f'{out_path}/roc_curve.png')
    plt.close()


def extract_results(idx_label_scores, T):
    y_true, y_pred, logit_scores_list, max_logit_scores_list, softmax_scores_list, max_softmax_scores_list = [], [], [], [], [], []
    for idx, label, scores in idx_label_scores:
        y_true.append(label)
        y_pred.append(np.argmax(scores, 0))
        logit_scores_list.append(scores)
        max_logit_scores_list.append(np.max(scores, 0))

        scores = torch.tensor(scores, dtype=torch.float)
        softmax_scores = F.softmax(scores / T, dim=0).numpy().tolist()
        softmax_scores_list.append(softmax_scores)
        max_softmax_scores_list.append(np.max(softmax_scores, 0))
    
    return y_true, y_pred, logit_scores_list, max_logit_scores_list, softmax_scores_list, max_softmax_scores_list


def plot_results(idx_label_scores, out_path, split: str = None, classes = None, n_classes: int = None, ood_idx_label_scores=None, T: int = 1):
    
    y_true_id, y_pred_id, logit_scores_id, softmax_scores_id, max_logit_scores_id, max_softmax_scores_id = extract_results(idx_label_scores, T)

    out_path = f'{out_path}/imgs'
    create_dirs_if_not_exist(out_path)

    if ood_idx_label_scores is None:
        confusion_matrix(y_true_id, y_pred_id, classes, out_path)

        metrics = {
            'accuracy': accuracy_score(y_true_id, y_pred_id),
            'precision': precision_score(y_true_id, y_pred_id, average='macro'),
            'recall': recall_score(y_true_id, y_pred_id, average='macro'),
            'fbeta': fbeta_score(y_true_id, y_pred_id, beta=1, average='macro'),
            'logit_auc_roc': roc_auc_score(y_true_id, max_logit_scores_id),
            'softmax_auc_roc': roc_auc_score(y_true_id, max_softmax_scores),
        }
        save_results(out_path, metrics, split, suffix='metrics')

    else:
        y_true_ood, y_pred_ood, logit_scores_ood, softmax_scores_ood, max_logit_scores_ood, max_softmax_scores_ood = extract_results(ood_idx_label_scores, T)

        plot_scores(max_logit_scores_id, max_logit_scores_ood, out_path, 'logit')
        plot_scores(max_softmax_scores_id, max_softmax_scores_ood, out_path, 'softmax')

        y_true = [[0] * len(max_logit_scores_id), [0] * len(max_logit_scores_ood)]
        max_logit_scores = max_logit_scores_id + max_logit_scores_ood
        max_softmax_scores = max_softmax_scores_id + max_softmax_scores_ood
        
        plot_curves(y_true, max_logit_scores, 'logit')
        plot_curves(y_true, max_softmax_scores, 'softmax')


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
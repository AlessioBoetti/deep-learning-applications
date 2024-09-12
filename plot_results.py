import sys
sys.path.insert(1, './src')
from utils import *
from plot_utils import *
import numpy as np


def main():
    # out_path = './Lab4 - Ex 1/results/OOD - CNN - 9 Conv Layers'
    # out_path = './Lab4 - Ex 2.1/results/Adv Images - CNN - 9 Conv Layers'
    # out_path = './Lab4 - Ex 2.2/results/Adv & OOD - CNN - 9 Conv Layers'
    # out_path = './Lab4 - Ex 3.1/results/Adv & OOD & Post - CNN - 9 Conv Layers'
    out_path = './Lab4 - Ex 3.1/results/OOD & Post - CNN - 9 Conv Layers'
    adv = False
    
    test_metrics = load_config(f'{out_path}/test_metrics.json')

    if adv:
        test_ood_metrics = load_config(f'{out_path}/test_adversarial_metrics.json')
    else:
        test_ood_metrics = load_config(f'{out_path}/test_ood_metrics.json')
    

    idx_label_scores = test_metrics['scores']
    ood_idx_label_scores = test_ood_metrics['scores']

    if adv:
        for alpha in np.arange(1, 11):
            plot_results(idx_label_scores, out_path, 'adv', ood_idx_label_scores=ood_idx_label_scores, eps=alpha)
    else:
        plot_results(idx_label_scores, out_path, 'ood', ood_idx_label_scores=ood_idx_label_scores)
    
    return


if __name__ == '__main__':
    main()
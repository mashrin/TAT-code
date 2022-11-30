import os
import torch
import pandas as pd
from .train_new import loss_metric, acc_metric, roc_auc_metric, f1_score_metric, ranked_acc_metric, bleu_metric, \
    kendall_tau_metric, compute_metric

metrics = {'loss': loss_metric, 'acc': acc_metric, 'auc': roc_auc_metric, 'macro_f1': f1_score_metric,
           'ranked_acc': ranked_acc_metric, 'bleu': bleu_metric, 'kendall_tau': kendall_tau_metric}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, 'checkpoint/2022_11_30_13_32_13/')

full_metrics = []
for epoch in range(5):
    predictions = torch.load(path + f'predictions_{epoch}.pt')
    labels = torch.load(path + f'labels_{epoch}.pt')
    results = compute_metric(predictions, labels, metrics)
    results['kendall_tau correlation'] = results['kendall_tau'].correlation
    results['kendall_tau (p-value)'] = results['kendall_tau'].pvalue
    del results['kendall_tau']
    full_metrics.append(results)

full_metrics = pd.DataFrame(full_metrics)
full_metrics.to_csv(path + 'results.csv', float_format='%.4f', index=True, index_label='epoch')

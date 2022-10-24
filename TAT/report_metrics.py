import pandas as pd
import numpy as np

models = ["gcn", "gat", "graphsage", "tagcn", "de-gnn", "tat"]
model_names = ["GCN", "GAT", "GraphSage", "TAGCN", "DE-GNN", "TAT"]

datasets = ["SMS-A", "CollegeMsg"]

for i, model in enumerate(models):

    print(model_names[i], end=' ')

    for dataset in datasets:
        res = pd.read_csv(f"./TAT/Results/{dataset}/{model}.csv")

        acc_idx = np.argmax(res["val_acc"])
        auc_idx = np.argmax(res["val_auc"])
        test_acc = res["test_acc"][acc_idx]
        test_auc = res["test_auc"][auc_idx]

        s = f"& {np.round(test_acc, 3): .3f} & {np.round(test_auc, 3): .3f} "

        if "test_bleu" in res.columns:
            idx = np.argmax(res["val_bleu"])
            test_val = res["test_bleu"][idx]
            s += f"& {np.round(test_val, 3): .3f}"

        if "test_kendall_tau" in res.columns:
            idx = np.argmax(res["val_kendall_tau"])
            test_val = res["test_kendall_tau"][idx]
            s += f"& {np.round(test_val, 3): .3f}"

        print(s, end=' ')
    print("\\\\")

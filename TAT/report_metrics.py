import pandas as pd
import numpy as np
import re

models = ["gcn", "gat", "graphsage", "tagcn", "de-gnn", "tat"]
model_names = ["GCN", "GAT", "GraphSage", "TAGCN", "DE-GNN", "TAT"]

datasets = ["SMS-A", "CollegeMsg"]

for i, model in enumerate(models):

    print(model_names[i], end=' ')

    for dataset in datasets:
        res = pd.read_csv(f"./TAT/checkpoint/{dataset}/{model}/record.csv")

        acc_idx = np.argmax(res["val_acc"])
        auc_idx = np.argmax(res["val_auc"])
        test_acc = res["test_acc"][acc_idx]
        test_auc = res["test_auc"][auc_idx]

        s = f"& {np.round(test_acc, 3): .3f} & {np.round(test_auc, 3): .3f} "

        if "test_bleu" in res.columns:
            idx = np.argmax(res["val_bleu"])
            test_val = res["test_bleu"][idx]
            s += f"& {np.round(test_val, 3): .3f} "

        if "test_kendall_tau" in res.columns:
            vals = res["val_kendall_tau"]
            vals = [re.findall(r'[\d]*[.][\d]+', v) for v in vals]
            for i in range(len(vals)):
                if len(vals[i]) == 0:
                    vals[i] = [0, 0]
                elif len(vals[i]) == 1:
                    print("Handle case")
                else:
                    vals[i] = [float(x) for x in vals[i]]
            vals = list(map(list, zip(*vals)))
            idx = np.argmax(vals[0])

            vals = res["test_kendall_tau"]
            vals = [re.findall(r'[\d]*[.][\d]+', v) for v in vals]
            for i in range(len(vals)):
                if len(vals[i]) == 0:
                    vals[i] = [0, 0]
                elif len(vals[i]) == 1:
                    print("Handle case")
                else:
                    vals[i] = [float(x) for x in vals[i]]
            vals = list(map(list, zip(*vals)))

            test_val = vals[0][idx]
            p_val = vals[1][idx]
            s += f"& {np.round(test_val, 3):.3f}({np.round(p_val, 3):.3f}) "

        print(s, end=' ')
    print("\\\\")

print("\n\n")

for i, model in enumerate(models):

    print(model_names[i], end=' ')
    res = pd.read_csv(f"./TAT/checkpoint/SMS-A-4/{model}/record.csv")

    acc_idx = np.argmax(res["val_acc"])
    auc_idx = np.argmax(res["val_auc"])
    test_acc = res["test_acc"][acc_idx]
    test_auc = res["test_auc"][auc_idx]

    s = f"& {np.round(test_acc, 3): .3f} & {np.round(test_auc, 3): .3f} "

    if "test_bleu" in res.columns:
        idx = np.argmax(res["val_bleu"])
        test_val = res["test_bleu"][idx]
        s += f"& {np.round(test_val, 3): .3f} "

    if "test_kendall_tau" in res.columns:
        vals = res["val_kendall_tau"]
        vals = [re.findall(r'[\d]*[.][\d]+', v) for v in vals]
        for i in range(len(vals)):
            if len(vals[i]) == 0:
                vals[i] = [0, 0]
            elif len(vals[i]) == 1:
                print("Handle case")
            else:
                vals[i] = [float(x) for x in vals[i]]
        vals = list(map(list, zip(*vals)))
        idx = np.argmax(vals[0])

        vals = res["test_kendall_tau"]
        vals = [re.findall(r'[\d]*[.][\d]+', v) for v in vals]
        for i in range(len(vals)):
            if len(vals[i]) == 0:
                vals[i] = [0, 0]
            elif len(vals[i]) == 1:
                print("Handle case")
            else:
                vals[i] = [float(x) for x in vals[i]]
        vals = list(map(list, zip(*vals)))

        test_val = vals[0][idx]
        p_val = vals[1][idx]
        s += f"& {np.round(test_val, 3):.3f}({np.round(p_val, 3):.3f}) "

    print(s, end=' ')
    print("\\\\")

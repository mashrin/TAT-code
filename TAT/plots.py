import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def make_plots():
    models = ["gcn", "gat", "graphsage", "tagcn", "de-gnn", "tat"]
    model_names = ["GCN", "GAT", "GraphSage", "TAGCN", "DE-GNN", "TAT"]

    datasets = ["SMS-A", "CollegeMsg"]

    metrics = [("Train loss", "train_loss"), ("Validation accuracy", "val_acc"), ("Validation AUC", "val_auc")]

    out_files = []

    for dataset in datasets:
        for metric in metrics:
            for i, model in enumerate(models):
                res = pd.read_csv(f"./TAT/Results/{dataset}/{model}.csv")
                plt.plot(res[metric[1]], label=model_names[i])

            plt.xlabel("Epoch")
            plt.ylabel(metric[0])
            plt.legend()
            filename = f"./TAT/figures/{dataset}_{metric[1]}.png"
            out_files.append(filename)
            plt.savefig(filename)
            plt.clf()

    return out_files


def merge_images(filenames, n_rows, n_cols, output_path):
    """
    Merge a list of image files into single image (for submission)

    Parameters:
        filenames (list of str): the images which are going to be merged
        n_rows (int): the number of rows in the image grid
        n_cols (int): the number of columns in the image grid
        output_path (str): the output path of the merged image
    """
    images = [Image.open(filename) for filename in filenames]
    width, height = images[0].size
    new_image = Image.new('RGB', (width * n_cols, height * n_rows))
    for row in range(n_rows):
        for col in range(n_cols):
            new_image.paste(
                images[row * n_cols + col],
                (width * col, height * row))

    new_image.save(output_path)


if __name__ == "__main__":
    filenames = make_plots()
    merge_images(filenames, 2, 3, "./TAT/figures/plots.png")
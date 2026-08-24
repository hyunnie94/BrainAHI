# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_attention_heatmap(attn_weights, title="Attention Map", save_path=None):
    plt.figure(figsize=(8,6))
    sns.heatmap(attn_weights, cmap="viridis")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Head")
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def summarize_classification_report(path):
    df = pd.read_csv(path)
    df = df[df["Label"].isin(["0", "1"])]
    print(df[["Label", "precision", "recall", "f1-score", "support"]])




    
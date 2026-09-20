"""Evaluation utilities shared by the training scripts."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, roc_curve,
)


def plot_confusion_matrix(cm, classes, title, output_path):
    fig, ax = plt.subplots()
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_score, title, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.05),
           xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def evaluate(y_true, y_pred, output_dir, model_name, y_score=None,
             classes=("REAL", "FAKE")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model_name.upper()} ===")
    labels = list(range(len(classes)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, zero_division=0
    )
    print(report)
    (output_dir / f"{model_name}_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix:")
    print(cm)
    plot_confusion_matrix(
        cm, classes, f"{model_name.upper()} - Confusion Matrix",
        output_dir / f"{model_name}_confusion_matrix.png",
    )

    if y_score is not None:
        auc_score = plot_roc_curve(
            y_true, y_score, f"{model_name.upper()} - ROC Curve",
            output_dir / f"{model_name}_roc_curve.png",
        )
        print(f"AUC: {auc_score:.4f}")

# batch/plots.py
# genera i grafici matplotlib (confusion matrix, threshold tuning, riepilogo)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt

def generate(tp, fp, fn, tn, thresholds, prec_scores, rec_scores, f1_scores, best_t, best_f1, accuracy, auc, rec1, prec1, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0",
    })

    # confusion matrix 
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i, j] > cm.max() / 2 else "#e0e0e0"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=c)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non Disastro (0)", "Disastro (1)"])
    ax.set_yticklabels(["Non Disastro (0)", "Disastro (1)"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(f"Confusion Matrix (soglia = {best_t:.2f})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"\n[Plot] confusion_matrix.png")

    # precision / recall / f1 vs threshold 
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, prec_scores, "o-", color="#00d2ff", lw=2, ms=7, label="Precision")
    ax.plot(thresholds, rec_scores, "s-", color="#ff6b6b", lw=2, ms=7, label="Recall")
    ax.plot(thresholds, f1_scores, "D-", color="#ffd93d", lw=2, ms=7, label="F1-Score")
    ax.axvline(x=best_t, color="#ffd93d", ls="--", alpha=0.7, lw=1.5)
    ax.annotate(
        f"Soglia: {best_t:.2f}", xy=(best_t, best_f1),
        xytext=(best_t + 0.05, best_f1 - 0.05),
        arrowprops=dict(arrowstyle="->", color="#ffd93d", lw=1.5),
        fontsize=11, color="#ffd93d", fontweight="bold",
    )
    ax.set_xlabel("Soglia")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Soglia (Classe Disastro)")
    ax.legend(loc="best", framealpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_tuning.png"), dpi=150)
    plt.close(fig)
    print("[Plot] threshold_tuning.png")

    # riepilogo metriche (bar chart) 
    names = ["Accuracy", "AUC", "Recall", "Precision", "F1"]
    vals = [accuracy, auc, rec1, prec1, best_f1]
    colors = ["#4ecdc4", "#45b7d1", "#ff6b6b", "#00d2ff", "#96ceb4"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, vals, color=colors, edgecolor="#e0e0e0", linewidth=0.5, height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax.set_xlim(0, 1.12)
    ax.set_title(f"Metriche — soglia {best_t:.2f}")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_summary.png"), dpi=150)
    plt.close(fig)
    print("[Plot] metrics_summary.png")

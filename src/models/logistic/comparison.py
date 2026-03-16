from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_comparison(results_raw, results_inference):
    """
    Plots the ROC curves for both the raw features model and the inference features model.
    """
    # How much is the imrovement? Is it worth it?
    diff = results_inference['auc'] - results_raw['auc']
    print(f"The difference in AUC is: {diff:.4f}")
    # in percentage
    diff_percentage = (diff / results_raw['auc']) * 100
    print(f"The improvement in AUC is: {diff_percentage:.2f}%")

    fpr_raw, tpr_raw, _ = roc_curve(results_raw["y_test"], results_raw["y_pred_proba"])
    fpr_inf, tpr_inf, _ = roc_curve(results_inference["y_test"], results_inference["y_pred_proba"])

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # ROC RAW FEATURES
    axes[0].plot(fpr_raw, tpr_raw, label=f"AUC = {results_raw['auc']:.3f}")
    axes[0].plot([0,1], [0,1], linestyle="--")
    axes[0].set_title("ROC Curve - Raw Features")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    # ROC INFERENCE FEATURES
    axes[1].plot(fpr_inf, tpr_inf, label=f"AUC = {results_inference['auc']:.3f}")
    axes[1].plot([0,1], [0,1], linestyle="--")
    axes[1].set_title("ROC Curve - Inference Features")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

import numpy as np
from sklearn.metrics import roc_curve

def tpr_at_fpr(
    y_true,
    y_probs,
    fpr_thresh=0.01
):
    
    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        if y_probs.ndim > 1:
            y_probs = y_probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        mask = fpr <= fpr_thresh
        return max(tpr[mask]) if mask.any() else 0.0

    tpr_scores = []
    for c in range(y_probs.shape[1]):
        y_true_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, c])
        mask = fpr <= fpr_thresh
        tpr_scores.append(max(tpr[mask]) if mask.any() else 0.0)
    return float(np.mean(tpr_scores))
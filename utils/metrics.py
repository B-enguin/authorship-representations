import torch

def f1(preds, labels):

    preds = preds.to(torch.int64)
    labels = labels.to(torch.int64)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1_score
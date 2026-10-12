"""Exact per-complex binary metrics, accounting for tied predictions."""
import torch

def contact_metrics(logits, labels, topk):
    scores = logits.detach().double().cpu().flatten()
    labels = labels.detach().double().cpu().flatten()
    order = scores.argsort(descending=True)
    scores, labels = scores[order], labels[order]
    positives = labels.sum().item()
    negatives = len(labels) - positives
    ends = torch.cat((torch.where(scores[:-1] != scores[1:])[0], torch.tensor([len(scores)-1])))
    tp = labels.cumsum(0)[ends]
    fp = (1-labels).cumsum(0)[ends]
    if positives:
        recall = tp / positives
        ap = ((recall - torch.cat((torch.zeros(1), recall[:-1]))) * tp / (ends+1)).sum().item()
    else:
        ap = None
    auc = None
    if positives and negatives:
        tpr = torch.cat((torch.zeros(1), tp/positives))
        fpr = torch.cat((torch.zeros(1), fp/negatives))
        auc = torch.trapz(tpr, fpr).item()
    k = min(topk, len(labels))
    return dict(contact_ap=ap, contact_auroc=auc,
                precision_at_k=labels[:k].mean().item(),
                recall_at_k=labels[:k].sum().item()/positives if positives else None,
                contact_pairs=int(positives), evaluated_pairs=len(labels), k=k)

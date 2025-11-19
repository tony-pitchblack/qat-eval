import torch


def ndcg_at_k(preds, targets, k=10):
    topk = torch.topk(preds, k=k, dim=1).indices
    targets = targets.unsqueeze(1)
    hits = (topk == targets).float()
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=preds.device).float())
    dcg = (hits * discounts).sum(dim=1)
    idcg = torch.ones_like(dcg)
    ndcg = dcg / idcg
    return ndcg.mean().item()


def accuracy(preds, targets):
    pred_labels = preds.argmax(dim=1)
    correct = (pred_labels == targets).float().mean()
    return correct.item()


def psnr(*args, **kwargs):
    raise NotImplementedError


def rocauc(*args, **kwargs):
    raise NotImplementedError
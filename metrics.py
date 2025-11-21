import torch
from functools import partial
from typing import Dict, List, Tuple, Callable


def ndcg_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    topk = torch.topk(preds, k=k, dim=1).indices
    targets = targets.unsqueeze(1)
    hits = (topk == targets).float()
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=preds.device).float())
    dcg = (hits * discounts).sum(dim=1)
    idcg = torch.ones_like(dcg)
    ndcg = dcg / idcg
    return ndcg.mean().item()


def hr_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    topk = torch.topk(preds, k=k, dim=1).indices
    targets = targets.unsqueeze(1)
    hits = (topk == targets).any(dim=1).float()
    return hits.mean().item()


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    pred_labels = preds.argmax(dim=1)
    correct = (pred_labels == targets).float().mean()
    return correct.item()


def psnr(*args, **kwargs):
    raise NotImplementedError


def rocauc(preds, targets):
    """Compute ROC AUC score for binary classification."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("sklearn is required for ROC AUC computation. Install with: pip install scikit-learn")

    # Convert logits to probabilities
    if preds.dim() > 1 and preds.size(1) > 1:
        probs = torch.softmax(preds, dim=1)[:, 1].detach().cpu().numpy()
    else:
        probs = torch.sigmoid(preds.squeeze()).detach().cpu().numpy()

    targets_np = targets.detach().cpu().numpy()
    return roc_auc_score(targets_np, probs)


MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


model_name_to_metrics: Dict[str, List[Tuple[str, MetricFn]]] = {
    "sasrec": [
        ("NDCG_at_5", partial(ndcg_at_k, k=5)),
        ("NDCG_at_10", partial(ndcg_at_k, k=10)),
        ("NDCG_at_20", partial(ndcg_at_k, k=20)),
        ("HR_at_5", partial(hr_at_k, k=5)),
        ("HR_at_10", partial(hr_at_k, k=10)),
        ("HR_at_20", partial(hr_at_k, k=20)),
    ],
    "simple_cnn": [
        ("accuracy", accuracy),
    ],
    "lstm": [
        ("ROCAUC", rocauc),
    ],
}

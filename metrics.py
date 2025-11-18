import torch

def ndcg_at_k(preds, targets, k=10):
    """
    Compute NDCG@k for batched predictions & targets.
    preds: Tensor of shape (batch_size, num_items), logits or scores.
    targets: Tensor of shape (batch_size,), contains ground-truth item indices.
    k: cutoff rank.
    
    Returns:
        mean ndcg@k over the batch (float)
    """
    topk = torch.topk(preds, k=k, dim=1).indices  # (batch_size, k)
    targets = targets.unsqueeze(1)  # (batch_size, 1)
    hits = (topk == targets).float()  # (batch_size, k)
    # Compute discounted gain
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=preds.device).float())
    dcg = (hits * discounts).sum(dim=1)
    # Ideal DCG is always 1 for single ground-truth if k >= 1, else 0
    idcg = torch.ones_like(dcg)
    ndcg = dcg / idcg
    return ndcg.mean().item()
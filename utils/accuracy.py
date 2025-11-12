import torch
import numpy as np


def evaluate_similarity(pred_similarity, target_similarity):

    batch_size = pred_similarity.shape[0]
    pred_similarity = pred_similarity.detach().cpu().numpy()
    target_similarity = target_similarity.detach().cpu().numpy()

    HR_at_k = [1, 5, 10, 20, 50]
    NDCG_at_k = [5, 10, 20, 50]
    HR_counts = {k: 0 for k in HR_at_k}
    ndcg_scores = {k: [] for k in NDCG_at_k}
    R5_at_20_count = 0
    R10_at_50_count = 0
    reciprocal_ranks = []

    for i in range(batch_size):
        # exclude query trajectory
        pred_rank = [idx for idx in np.argsort(-pred_similarity[i]) if idx != i]
        true_rank = [idx for idx in np.argsort(-target_similarity[i]) if idx != i]

        gt_idx = true_rank[0]
        rank = pred_rank.index(gt_idx) + 1
        reciprocal_ranks.append(1.0 / rank)

        for k in HR_at_k:
            pred_topk = set(pred_rank[:k])
            true_topk = set(true_rank[:k])
            hits = len(pred_topk & true_topk)
            HR_counts[k] += hits

        pred_top20 = set(pred_rank[:20])
        true_top5 = set(true_rank[:5])
        R5_at_20_count += len(pred_top20 & true_top5)

        pred_top50 = set(pred_rank[:50])
        true_top10 = set(true_rank[:10])
        R10_at_50_count += len(pred_top50 & true_top10)

        # NDCG@k
        for k in NDCG_at_k:

            pred_topk = pred_rank[:k]
            true_topk = set(true_rank[:k])
            rels = np.array([1 if idx in true_topk else 0 for idx in pred_topk])

            # DCG
            dcg = np.sum(rels / np.log2(np.arange(2, k + 2)))
            # IDCG
            idcg = np.sum(np.ones_like(rels) / np.log2(np.arange(2, k + 2)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores[k].append(ndcg)

    metrics = {f"HR@{k}": HR_counts[k] / (batch_size * k) for k in HR_at_k}
    metrics["R5@20"] = R5_at_20_count / (batch_size * 5)
    metrics["R10@50"] = R10_at_50_count / (batch_size * 10)
    metrics["MRR"] = np.mean(reciprocal_ranks)
    for k in NDCG_at_k:
        metrics[f"NDCG@{k}"] = np.mean(ndcg_scores[k])

    return metrics

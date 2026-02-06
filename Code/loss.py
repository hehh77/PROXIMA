import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def masked_mse_loss(preds, target, mask):
    """
    Computes Mean Squared Error loss only on masked elements.
    Used for Masked Expression Prediction (MEP) and Neighborhood Expression Imputation (NEI).
    """
    if not mask.any():
        return torch.tensor(0.0, device=preds.device)
    return F.mse_loss(preds[mask], target[mask])

def spatially_aware_contrastive_loss(
    features, 
    patch_membership, 
    coordinates, 
    temperature=0.2, 
    k_negatives=64
):
    """
    Implements Spatially-Aware Contrastive Learning (SCL) loss.
    Attracts cells from the same patch and repels spatially distant cells.
    """
    with autocast(enabled=False):
        features = features.float()
        patch_membership = patch_membership.float()
        coordinates = coordinates.float()

        if torch.isnan(coordinates).any() or torch.isnan(features).any():
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print("WARNING: NaN in inputs for SCL loss. Skipping.")
            return torch.tensor(0.0, device=features.device)

        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # Filter zero-norm features
        norms = torch.norm(features, p=2, dim=1)
        valid_norm_mask = norms > 1e-6
        if not valid_norm_mask.all():
            features = features[valid_norm_mask]
            patch_membership = patch_membership[valid_norm_mask]
            coordinates = coordinates[valid_norm_mask]
            batch_size = features.shape[0]
            if batch_size <= 1:
                return torch.tensor(0.0, device=device)

        features_norm = F.normalize(features, p=2, dim=1) + 1e-10
        similarity_matrix = torch.matmul(features_norm, features_norm.T)

        # Positive pairs: cells in the same patch
        positives_mask = (patch_membership.unsqueeze(1) == patch_membership.unsqueeze(0))
        positives_mask.fill_diagonal_(False)

        potential_negatives_mask = ~positives_mask
        potential_negatives_mask.fill_diagonal_(False)
        
        # Select hard negatives based on spatial distance (furthest cells)
        dist_matrix = torch.cdist(coordinates, coordinates, p=2)
        if torch.isnan(dist_matrix).any():
            return torch.tensor(0.0, device=device)

        dist_matrix_for_negs = dist_matrix.clone()
        dist_matrix_for_negs[~potential_negatives_mask] = -1.0
        
        max_possible_k = potential_negatives_mask.sum(dim=1).min()
        if max_possible_k.item() < 1:
            return torch.tensor(0.0, device=device)
        actual_k = min(k_negatives, int(max_possible_k.item()))

        _, topk_indices = torch.topk(dist_matrix_for_negs, k=actual_k, dim=1, largest=True)
        
        negatives_mask = torch.zeros_like(potential_negatives_mask, dtype=torch.bool)
        negatives_mask.scatter_(1, topk_indices, True)

        valid_anchors_mask = (positives_mask.sum(dim=1) > 0) & (negatives_mask.sum(dim=1) > 0)
        if not valid_anchors_mask.any():
            return torch.tensor(0.0, device=device)

        similarity_matrix_valid = similarity_matrix[valid_anchors_mask]
        positives_mask_valid = positives_mask[valid_anchors_mask]
        negatives_mask_valid = negatives_mask[valid_anchors_mask]

        logits = similarity_matrix_valid / temperature
        
        mask_pos_and_neg = positives_mask_valid | negatives_mask_valid
        logits_masked = logits.clone()
        logits_masked[~mask_pos_and_neg] = float('-inf')

        log_probs = F.log_softmax(logits, dim=1, dtype=torch.float32)
        sum_log_probs_pos = (log_probs * positives_mask_valid.float()).sum(1)
        num_positives = torch.clamp(positives_mask_valid.sum(1), min=1e-8)
        mean_log_prob_pos = sum_log_probs_pos / num_positives
        
        valid_mean_mask = ~(torch.isnan(mean_log_prob_pos) | torch.isinf(mean_log_prob_pos))
        if not valid_mean_mask.any():
            return torch.tensor(1e-5, device=device)
        mean_log_prob_pos = mean_log_prob_pos[valid_mean_mask]

        loss = -mean_log_prob_pos.mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1e-5, device=device)

    return loss

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, requires_grad=True)
    else:
        return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

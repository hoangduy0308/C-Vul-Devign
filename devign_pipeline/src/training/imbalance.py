"""Imbalance handling utilities for vulnerability detection training."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from typing import Tuple, Optional


def compute_class_weights(labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        labels: Binary labels array (0=non-vuln, 1=vuln)
    
    Returns:
        (weight_class_0, weight_class_1)
    """
    n_samples = len(labels)
    n_pos = np.sum(labels == 1)
    n_neg = n_samples - n_pos
    
    # Inverse frequency weighting
    weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
    weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
    
    return weight_neg, weight_pos


def get_pos_weight(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    
    pos_weight = n_neg / n_pos
    
    Args:
        labels: Binary labels array
    
    Returns:
        pos_weight scalar for BCE loss
    """
    n_pos = np.sum(labels == 1)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0:
        return 1.0
    
    return n_neg / n_pos


def get_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a weighted sampler for balanced batches.
    
    Args:
        labels: Binary labels array
    
    Returns:
        WeightedRandomSampler that samples classes equally
    """
    class_weights = compute_class_weights(labels)
    sample_weights = np.array([class_weights[int(l)] for l in labels])
    
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold on validation set.
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        metric: 'f1' or 'mcc'
    
    Returns:
        (optimal_threshold, best_score)
    """
    from sklearn.metrics import f1_score, matthews_corrcoef
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for imbalanced classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (gamma=0 is standard CE)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


def print_class_distribution(labels: np.ndarray, split_name: str = ""):
    """Print class distribution statistics."""
    n_total = len(labels)
    n_pos = np.sum(labels == 1)
    n_neg = n_total - n_pos
    ratio = n_neg / n_pos if n_pos > 0 else float('inf')
    
    print(f"{split_name} Class Distribution:")
    print(f"  Total: {n_total}")
    print(f"  Non-vulnerable (0): {n_neg} ({100*n_neg/n_total:.1f}%)")
    print(f"  Vulnerable (1): {n_pos} ({100*n_pos/n_total:.1f}%)")
    print(f"  Imbalance ratio: {ratio:.2f}:1")
    print(f"  Recommended pos_weight: {ratio:.2f}")

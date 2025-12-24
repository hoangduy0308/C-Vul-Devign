"""
Simplified Training Script for Devign Vulnerability Detection.

Based on Oracle recommendations:
- Single loss: BCEWithLogitsLoss(pos_weight)
- Multi-seed evaluation with mean ± std
- Clean ablation-ready design

Usage:
    python train_simplified.py --config baseline --seeds 3
    python train_simplified.py --config precision_focused --seeds 5
    python train_simplified.py --ablation  # Run full ablation study
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config_simplified import (
    BaselineConfig,
    get_baseline_config,
    get_recall_focused_config,
    get_precision_focused_config,
    get_large_config,
    create_ablation_configs,
    get_seeds_for_evaluation,
    AblationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= LOSS FUNCTION =============

class SimplifiedLoss(nn.Module):
    """
    Simplified loss function following Oracle recommendations.
    
    Options:
    1. bce_weighted: BCEWithLogitsLoss with pos_weight (RECOMMENDED)
    2. focal_only: FocalLoss without class weight stacking
    
    DO NOT stack multiple loss modifications together.
    """
    
    def __init__(
        self,
        loss_type: str = "bce_weighted",
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        
        if loss_type == "bce_weighted":
            # Oracle recommended: simple BCE with pos_weight
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        elif loss_type == "focal_only":
            # Alternative: Focal loss without additional weighting
            self.criterion = None  # Computed manually
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: [B, 2] or [B, 1] model output (before softmax/sigmoid)
            targets: [B] target labels (0 or 1)
        """
        # Handle 2-class output
        if logits.dim() == 2 and logits.size(1) == 2:
            # Use class 1 (vulnerable) logit
            logits = logits[:, 1]
        elif logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        
        targets = targets.float()
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        if self.loss_type == "bce_weighted":
            return self.criterion(logits, targets)
        
        elif self.loss_type == "focal_only":
            # Focal loss: FL(p_t) = -(1-p_t)^gamma * log(p_t)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** self.focal_gamma
            return (focal_weight * bce_loss).mean()
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


def compute_pos_weight(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    
    pos_weight = n_negative / n_positive
    
    This balances the loss contribution of positive and negative classes.
    """
    n_positive = (labels == 1).sum()
    n_negative = (labels == 0).sum()
    
    if n_positive == 0:
        logger.warning("No positive samples! Using pos_weight=1.0")
        return 1.0
    
    pos_weight = n_negative / n_positive
    logger.info(f"Class distribution: {n_negative} neg, {n_positive} pos")
    logger.info(f"Computed pos_weight: {pos_weight:.4f}")
    
    return float(pos_weight)


# ============= METRICS =============

@dataclass
class EvalMetrics:
    """Evaluation metrics for a single run."""
    f1: float
    precision: float
    recall: float
    auc: float
    accuracy: float
    threshold: float
    
    def to_dict(self) -> Dict:
        return {
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'auc': self.auc,
            'accuracy': self.accuracy,
            'threshold': self.threshold,
        }


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold_min: float = 0.3,
    threshold_max: float = 0.7,
    threshold_step: float = 0.01,
) -> EvalMetrics:
    """
    Compute evaluation metrics with optimal threshold search.
    
    Oracle: Focus on F1 and PR-AUC, not just ROC-AUC.
    """
    from sklearn.metrics import (
        precision_recall_curve,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
    
    # Find optimal threshold for F1
    best_f1 = 0.0
    best_threshold = 0.5
    
    for thresh in np.arange(threshold_min, threshold_max + threshold_step, threshold_step):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Compute metrics at optimal threshold
    preds = (probs >= best_threshold).astype(int)
    
    return EvalMetrics(
        f1=f1_score(labels, preds, zero_division=0),
        precision=precision_score(labels, preds, zero_division=0),
        recall=recall_score(labels, preds, zero_division=0),
        auc=roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        accuracy=accuracy_score(labels, preds),
        threshold=best_threshold,
    )


# ============= MULTI-SEED EVALUATION =============

@dataclass
class AggregatedResults:
    """Aggregated results across multiple seeds."""
    config_name: str
    n_seeds: int
    
    f1_mean: float
    f1_std: float
    
    precision_mean: float
    precision_std: float
    
    recall_mean: float
    recall_std: float
    
    auc_mean: float
    auc_std: float
    
    threshold_mean: float
    
    def __str__(self) -> str:
        return (
            f"{self.config_name} ({self.n_seeds} seeds):\n"
            f"  F1:        {self.f1_mean:.4f} ± {self.f1_std:.4f}\n"
            f"  Precision: {self.precision_mean:.4f} ± {self.precision_std:.4f}\n"
            f"  Recall:    {self.recall_mean:.4f} ± {self.recall_std:.4f}\n"
            f"  AUC:       {self.auc_mean:.4f} ± {self.auc_std:.4f}\n"
            f"  Threshold: {self.threshold_mean:.4f}"
        )


def aggregate_results(results: List[EvalMetrics], config_name: str) -> AggregatedResults:
    """
    Aggregate results from multiple seeds.
    
    Oracle: Report mean ± std across seeds.
    """
    f1s = [r.f1 for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    aucs = [r.auc for r in results]
    thresholds = [r.threshold for r in results]
    
    return AggregatedResults(
        config_name=config_name,
        n_seeds=len(results),
        f1_mean=np.mean(f1s),
        f1_std=np.std(f1s),
        precision_mean=np.mean(precisions),
        precision_std=np.std(precisions),
        recall_mean=np.mean(recalls),
        recall_std=np.std(recalls),
        auc_mean=np.mean(aucs),
        auc_std=np.std(aucs),
        threshold_mean=np.mean(thresholds),
    )


# ============= ABLATION STUDY =============

def run_ablation_study(
    data_dir: str,
    output_dir: str,
    n_seeds: int = 3,
    device: str = "cuda",
) -> Dict[str, AggregatedResults]:
    """
    Run systematic ablation study.
    
    Oracle: Change ONE thing at a time, report mean ± std.
    """
    ablation_configs = create_ablation_configs()
    all_results = {}
    
    logger.info(f"Running ablation study with {len(ablation_configs)} configs, {n_seeds} seeds each")
    
    for config_name, config in ablation_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {config_name}")
        logger.info(f"{'='*60}")
        
        seed_results = []
        seeds = get_seeds_for_evaluation(n_seeds, config.ensemble_base_seed)
        
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            
            # TODO: Call actual training function here
            # metrics = train_single_run(config, seed, data_dir, device)
            # seed_results.append(metrics)
            
            # Placeholder for demo
            logger.info(f"    (Training would run here)")
        
        if seed_results:
            aggregated = aggregate_results(seed_results, config_name)
            all_results[config_name] = aggregated
            logger.info(f"\n{aggregated}")
    
    return all_results


def save_ablation_results(
    results: Dict[str, AggregatedResults],
    output_path: str,
):
    """Save ablation results to JSON for analysis."""
    data = {}
    for name, res in results.items():
        data[name] = {
            'n_seeds': res.n_seeds,
            'f1_mean': res.f1_mean,
            'f1_std': res.f1_std,
            'precision_mean': res.precision_mean,
            'precision_std': res.precision_std,
            'recall_mean': res.recall_mean,
            'recall_std': res.recall_std,
            'auc_mean': res.auc_mean,
            'auc_std': res.auc_std,
            'threshold_mean': res.threshold_mean,
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved ablation results to {output_path}")

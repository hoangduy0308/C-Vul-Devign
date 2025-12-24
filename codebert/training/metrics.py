"""Metrics computation for CodeBERT vulnerability detection."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "threshold": self.threshold,
        }
    
    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f} | "
            f"Precision: {self.precision:.4f} | "
            f"Recall: {self.recall:.4f} | "
            f"F1: {self.f1:.4f} | "
            f"ROC-AUC: {self.roc_auc:.4f} | "
            f"PR-AUC: {self.pr_auc:.4f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> MetricsResult:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_proba: Predicted probabilities for positive class
        threshold: Classification threshold
    
    Returns:
        MetricsResult containing all metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = 0.0
    
    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        pr_auc = 0.0
    
    return MetricsResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        threshold=threshold,
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    step: float = 0.01,
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for F1 score.
    
    Args:
        y_true: Ground truth binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        min_threshold: Minimum threshold to search
        max_threshold: Maximum threshold to search
        step: Step size for threshold search
    
    Returns:
        Tuple of (best_threshold, best_score)
    """
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in np.arange(min_threshold, max_threshold + step, step):
        y_pred = (y_proba >= thresh).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


@dataclass
class MetricsTracker:
    """Track and log training metrics across epochs."""
    
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[MetricsResult] = field(default_factory=list)
    best_f1: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_result: MetricsResult,
    ) -> bool:
        """
        Update tracker with epoch results.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for epoch
            val_loss: Validation loss for epoch
            val_result: Validation metrics
        
        Returns:
            True if this is a new best model
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_metrics.append(val_result)
        
        is_best = val_result.f1 > self.best_f1
        if is_best:
            self.best_f1 = val_result.f1
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return is_best
    
    def should_stop(self, patience: int) -> bool:
        """Check if early stopping should trigger."""
        return self.patience_counter >= patience
    
    def get_summary(self) -> Dict:
        """Get summary of training history."""
        return {
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "final_val_loss": self.val_losses[-1] if self.val_losses else 0.0,
            "total_epochs": len(self.train_losses),
        }
    
    def log_epoch(self, epoch: int, total_epochs: int) -> str:
        """Generate log string for current epoch."""
        if not self.train_losses:
            return ""
        
        train_loss = self.train_losses[-1]
        val_loss = self.val_losses[-1]
        val_result = self.val_metrics[-1]
        
        return (
            f"Epoch {epoch}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"{val_result}"
        )

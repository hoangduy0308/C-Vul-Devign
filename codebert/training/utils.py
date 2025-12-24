"""Training utilities for CodeBERT vulnerability detection."""

import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    no_decay_params: tuple = ("bias", "LayerNorm.weight"),
) -> AdamW:
    """
    Create AdamW optimizer with weight decay fix.
    
    Args:
        model: The model to optimize
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        no_decay_params: Parameter names that should not have weight decay
    
    Returns:
        Configured AdamW optimizer
    """
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_params) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_params) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    return AdamW(optimizer_grouped_parameters, lr=learning_rate)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create linear warmup scheduler with linear decay.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
    
    Returns:
        LambdaLR scheduler with linear warmup and decay
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def compute_class_weights(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    
    pos_weight = n_negative / n_positive
    
    Args:
        labels: Binary labels array (0=non-vuln, 1=vuln)
    
    Returns:
        pos_weight value for BCEWithLogitsLoss
    """
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)
    
    if n_positive == 0:
        return 1.0
    
    return float(n_negative / n_positive)


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    
    for i in range(info["device_count"]):
        info["devices"].append({
            "name": torch.cuda.get_device_name(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory / 1e9,
        })
    
    return info

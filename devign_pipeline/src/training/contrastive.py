"""
Contrastive Learning Module for Vulnerability Detection.

Implements:
1. Supervised Contrastive Loss (SupCon) - for labeled data
2. SimCLR-style loss - for self-supervised learning
3. Code-specific augmentation strategies
4. Combined training pipeline (BCE + Contrastive)

Key insight: Contrastive learning helps learn better representations by:
- Pulling same-class samples (both vulnerable or both safe) closer
- Pushing different-class samples apart in embedding space
- Creating more discriminative features for vulnerability detection

Reference Papers:
- "Supervised Contrastive Learning" (Khosla et al., 2020)
- "A Simple Framework for Contrastive Learning" (SimCLR, Chen et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


# ============= CONTRASTIVE LOSSES =============

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    For vulnerability detection:
    - Positive pairs: samples with same label (both vuln or both safe)
    - Negative pairs: samples with different labels
    
    This encourages the model to cluster vulnerable code together
    and safe code together in the embedding space.
    
    Formula:
        L = -sum_i sum_{p in P(i)} log(exp(z_i·z_p/τ) / sum_{a in A(i)} exp(z_i·z_a/τ))
    
    Args:
        temperature: Temperature scaling factor (default: 0.07)
        contrast_mode: 'all' or 'one' (use all positives or just one anchor)
        base_temperature: Base temperature for scaling
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        contrast_mode: str = 'all',
        base_temperature: float = None  # If None, use temperature (no scaling)
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature if base_temperature is not None else temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Hidden representations [B, D] or [B, N_views, D]
            labels: Ground truth labels [B]
            mask: Optional contrastive mask [B, B]
            
        Returns:
            Contrastive loss scalar
        """
        device = features.device
        
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B, 1, D]
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # Normalize features
        features = F.normalize(features, dim=2)
        
        # Reshape for contrast
        contrast_features = features.reshape(batch_size * n_views, -1)  # [B*N, D]
        
        if self.contrast_mode == 'one':
            anchor_features = features[:, 0]  # [B, D]
            anchor_count = 1
        else:  # 'all'
            anchor_features = contrast_features
            anchor_count = n_views
        
        # Create label mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)  # [B, 1]
        if mask is None:
            # [B, B] - entry (i,j) = 1 if labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Expand mask for multiple views
        if n_views > 1:
            mask = mask.repeat(anchor_count, n_views)  # [B*anchor, B*n_views]
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.matmul(anchor_features, contrast_features.T)  # [B*anchor, B*n_views]
        anchor_dot_contrast = anchor_dot_contrast / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create self-contrast mask (exclude self from denominator)
        if n_views > 1:
            # For multi-view, self-mask is more complex
            logits_mask = torch.ones_like(mask)
            logits_mask = logits_mask.scatter_(
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
        else:
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        
        # Mask out self-contrast
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero when no positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class SimCLRLoss(nn.Module):
    """
    SimCLR-style Contrastive Loss (NT-Xent).
    
    Self-supervised version where positive pairs come from augmentations
    of the same sample, not from labels.
    
    Args:
        temperature: Temperature scaling factor
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.
        
        Args:
            z_i: Features from first augmentation [B, D]
            z_j: Features from second augmentation [B, D]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        
        # Create positive pair mask
        # Positives are at positions (i, i+B) and (i+B, i)
        sim_ij = torch.diag(sim_matrix, batch_size)  # [B]
        sim_ji = torch.diag(sim_matrix, -batch_size)  # [B]
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # [2B]
        
        # Create negative mask (exclude self and positive pair)
        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        mask = mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False
        
        negatives = sim_matrix[mask].reshape(2 * batch_size, -1)  # [2B, 2B-2]
        
        # Compute loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)  # [2B, 2B-1]
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)  # Positive is always at index 0
        
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss


class CombinedContrastiveLoss(nn.Module):
    """
    Combined loss for vulnerability detection with contrastive learning.
    
    Total Loss = λ_cls * Classification_Loss + λ_con * Contrastive_Loss
    
    Args:
        cls_weight: Weight for classification loss (default: 1.0)
        con_weight: Weight for contrastive loss (default: 0.5)
        temperature: Contrastive temperature
        use_supcon: Use supervised contrastive (True) or SimCLR (False)
        pos_weight: Positive class weight for BCE loss
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        con_weight: float = 0.5,
        temperature: float = 0.07,
        use_supcon: bool = True,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.con_weight = con_weight
        self.use_supcon = use_supcon
        
        # Classification loss
        self.cls_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.label_smoothing = label_smoothing
        
        # Contrastive loss
        if use_supcon:
            self.con_criterion = SupConLoss(temperature=temperature)
        else:
            self.con_criterion = SimCLRLoss(temperature=temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        embeddings_aug: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits [B, 2] or [B, 1]
            embeddings: Feature embeddings [B, D]
            labels: Ground truth labels [B]
            embeddings_aug: Augmented embeddings for SimCLR [B, D]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss values for logging
        """
        # Classification loss
        if logits.dim() == 2 and logits.size(1) == 2:
            cls_logits = logits[:, 1]
        else:
            cls_logits = logits.squeeze()
        
        targets = labels.float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        cls_loss = self.cls_criterion(cls_logits, targets)
        
        # Contrastive loss
        if self.use_supcon:
            con_loss = self.con_criterion(embeddings, labels)
        else:
            if embeddings_aug is None:
                raise ValueError("embeddings_aug required for SimCLR mode")
            con_loss = self.con_criterion(embeddings, embeddings_aug)
        
        # Combined loss
        total_loss = self.cls_weight * cls_loss + self.con_weight * con_loss
        
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'con_loss': con_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============= CODE AUGMENTATION STRATEGIES =============

class CodeAugmentor:
    """
    Code-specific augmentation strategies for contrastive learning.
    
    Augmentation strategies designed to preserve semantic meaning
    while creating different views of the same code:
    1. Token dropout - randomly drop tokens
    2. Token masking - replace with [MASK]
    3. Token shuffling - shuffle within small windows (DISABLED by default)
    
    WARNING: Token shuffling breaks token_type_ids alignment!
    Token type IDs are position-dependent. If you shuffle tokens without
    also shuffling their corresponding type IDs, the embeddings will be
    semantically incorrect. Only enable shuffle_prob if you're NOT using
    extended_token_type_ids, or if you implement synchronized shuffling.
    """
    
    def __init__(
        self,
        dropout_prob: float = 0.1,
        mask_prob: float = 0.1,
        shuffle_prob: float = 0.0,  # Disabled by default - breaks token_type_ids
        shuffle_window: int = 3,
        mask_token_id: int = 1,
        pad_token_id: int = 0,
        special_token_ids: Optional[List[int]] = None
    ):
        self.dropout_prob = dropout_prob
        self.mask_prob = mask_prob
        self.shuffle_prob = shuffle_prob
        self.shuffle_window = shuffle_window
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = special_token_ids or [0, 1, 2, 3, 4]  # PAD, UNK, CLS, EOS, SEP
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random augmentation to input."""
        aug_ids = input_ids.clone()
        B, L = input_ids.shape
        device = input_ids.device
        
        # Mask of valid tokens (not padding, not special)
        valid_mask = (attention_mask > 0)
        for special_id in self.special_token_ids:
            valid_mask = valid_mask & (input_ids != special_id)
        
        # Token dropout
        if self.dropout_prob > 0:
            dropout_mask = torch.rand(B, L, device=device) < self.dropout_prob
            dropout_mask = dropout_mask & valid_mask
            aug_ids = aug_ids.masked_fill(dropout_mask, self.pad_token_id)
        
        # Token masking
        if self.mask_prob > 0:
            mask_mask = torch.rand(B, L, device=device) < self.mask_prob
            mask_mask = mask_mask & valid_mask & (aug_ids != self.pad_token_id)
            aug_ids = aug_ids.masked_fill(mask_mask, self.mask_token_id)
        
        # Token shuffling (within windows)
        if self.shuffle_prob > 0:
            aug_ids = self._window_shuffle(aug_ids, valid_mask)
        
        return aug_ids
    
    def _window_shuffle(
        self,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Shuffle tokens within small windows."""
        B, L = input_ids.shape
        device = input_ids.device
        
        for b in range(B):
            if torch.rand(1).item() > self.shuffle_prob:
                continue
            
            valid_indices = valid_mask[b].nonzero().squeeze(-1)
            if len(valid_indices) < self.shuffle_window:
                continue
            
            # Pick a random starting position
            start_idx = torch.randint(0, len(valid_indices) - self.shuffle_window + 1, (1,)).item()
            window_indices = valid_indices[start_idx:start_idx + self.shuffle_window]
            
            # Shuffle the window
            perm = torch.randperm(self.shuffle_window)
            shuffled_tokens = input_ids[b, window_indices[perm]]
            input_ids[b, window_indices] = shuffled_tokens
        
        return input_ids
    
    def generate_views(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_views: int = 2
    ) -> List[torch.Tensor]:
        """Generate multiple augmented views of the input."""
        views = []
        for _ in range(n_views):
            aug_ids = self(input_ids, attention_mask)
            views.append(aug_ids)
        return views


# ============= PROJECTION HEAD =============

class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps the encoder output to a lower-dimensional space
    where contrastive loss is computed.
    
    Architecture: Linear -> BN -> ReLU -> Linear
    
    Args:
        input_dim: Input dimension (encoder output)
        hidden_dim: Hidden layer dimension
        output_dim: Output projection dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ============= CONTRASTIVE CONFIG =============

@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    
    # Contrastive learning settings
    use_contrastive: bool = True
    contrastive_weight: float = 0.5  # λ_con in combined loss
    temperature: float = 0.07  # Lower = sharper distribution
    use_supcon: bool = True  # True for SupCon, False for SimCLR
    
    # Projection head
    projection_hidden_dim: int = 256
    projection_output_dim: int = 128
    projection_dropout: float = 0.1
    
    # Augmentation for SimCLR
    # NOTE: shuffle_prob=0 by default because shuffling breaks token_type_ids alignment
    # Token type IDs are position-dependent; shuffling tokens without shuffling
    # their type IDs causes semantic mismatch in embeddings
    aug_dropout_prob: float = 0.1
    aug_mask_prob: float = 0.1
    aug_shuffle_prob: float = 0.0  # Disabled: breaks token_type_ids alignment
    aug_shuffle_window: int = 3
    
    # Warmup: start with pure classification, then add contrastive
    contrastive_warmup_epochs: int = 2
    
    # Curriculum: increase contrastive weight over time
    use_curriculum: bool = False
    curriculum_start_weight: float = 0.1
    curriculum_max_weight: float = 0.5
    curriculum_epochs: int = 10
    
    def get_contrastive_weight(self, epoch: int) -> float:
        """Get contrastive weight based on curriculum."""
        if epoch < self.contrastive_warmup_epochs:
            return 0.0
        
        if not self.use_curriculum:
            return self.contrastive_weight
        
        # Linear curriculum
        curriculum_epoch = epoch - self.contrastive_warmup_epochs
        if curriculum_epoch >= self.curriculum_epochs:
            return self.curriculum_max_weight
        
        progress = curriculum_epoch / self.curriculum_epochs
        return self.curriculum_start_weight + progress * (
            self.curriculum_max_weight - self.curriculum_start_weight
        )


# ============= TRAINING UTILITIES =============

def compute_contrastive_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics to evaluate contrastive learning quality.
    
    Metrics:
    - intra_class_sim: Average similarity within same class
    - inter_class_sim: Average similarity between different classes
    - alignment: Measures if same-class samples are similar
    - uniformity: Measures if embeddings are uniformly distributed
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Create masks
    labels = labels.view(-1, 1)
    same_class_mask = (labels == labels.T).float()
    diff_class_mask = 1 - same_class_mask
    
    # Remove diagonal
    eye = torch.eye(len(labels), device=embeddings.device)
    same_class_mask = same_class_mask - eye
    
    # Compute metrics
    intra_class_sim = (sim_matrix * same_class_mask).sum() / (same_class_mask.sum() + 1e-8)
    inter_class_sim = (sim_matrix * diff_class_mask).sum() / (diff_class_mask.sum() + 1e-8)
    
    # Alignment: average pairwise distance for same-class samples
    alignment = -intra_class_sim  # Lower is better (closer samples)
    
    # Uniformity: measures how uniformly distributed embeddings are
    # Lower is better (more uniform)
    sq_pdist = (2 - 2 * sim_matrix).clamp(min=1e-8)
    uniformity = torch.log(torch.exp(-2 * sq_pdist).mean() + 1e-8)
    
    return {
        'intra_class_sim': intra_class_sim.item(),
        'inter_class_sim': inter_class_sim.item(),
        'sim_gap': (intra_class_sim - inter_class_sim).item(),
        'alignment': alignment.item(),
        'uniformity': uniformity.item()
    }


class ContrastiveTrainingCallback:
    """
    Callback for contrastive learning training loop.
    
    Handles:
    - Curriculum learning for contrastive weight
    - Logging contrastive metrics
    - Early stopping based on contrastive quality
    """
    
    def __init__(self, config: ContrastiveConfig):
        self.config = config
        self.history = []
    
    def on_epoch_start(self, epoch: int) -> float:
        """Return contrastive weight for this epoch."""
        return self.config.get_contrastive_weight(epoch)
    
    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        loss_dict: Dict[str, float],
        embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """Log batch metrics."""
        if embeddings is not None and labels is not None:
            metrics = compute_contrastive_metrics(embeddings, labels)
            loss_dict.update(metrics)
        
        self.history.append({
            'epoch': epoch,
            'batch': batch_idx,
            **loss_dict
        })
    
    def get_summary(self) -> Dict[str, float]:
        """Get training summary."""
        if not self.history:
            return {}
        
        summary = {}
        for key in self.history[-1].keys():
            if key not in ['epoch', 'batch']:
                values = [h[key] for h in self.history if key in h]
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
        
        return summary


if __name__ == "__main__":
    # Demo usage
    print("=== Contrastive Learning Module Demo ===\n")
    
    # Create dummy data
    batch_size = 8
    seq_len = 128
    hidden_dim = 256
    
    embeddings = torch.randn(batch_size, hidden_dim)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    
    # Test SupCon loss
    supcon_loss = SupConLoss(temperature=0.07)
    loss = supcon_loss(embeddings, labels)
    print(f"SupCon Loss: {loss.item():.4f}")
    
    # Test SimCLR loss
    simclr_loss = SimCLRLoss(temperature=0.5)
    z_i = torch.randn(batch_size, hidden_dim)
    z_j = torch.randn(batch_size, hidden_dim)
    loss = simclr_loss(z_i, z_j)
    print(f"SimCLR Loss: {loss.item():.4f}")
    
    # Test combined loss
    logits = torch.randn(batch_size, 2)
    combined_loss = CombinedContrastiveLoss(
        cls_weight=1.0,
        con_weight=0.5,
        temperature=0.07,
        use_supcon=True
    )
    total_loss, loss_dict = combined_loss(logits, embeddings, labels)
    print(f"\nCombined Loss: {loss_dict}")
    
    # Test contrastive metrics
    metrics = compute_contrastive_metrics(embeddings, labels)
    print(f"\nContrastive Metrics: {metrics}")
    
    # Test code augmentor
    input_ids = torch.randint(5, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -20:] = 0  # Simulate padding
    
    augmentor = CodeAugmentor(dropout_prob=0.1, mask_prob=0.1)
    aug_ids = augmentor(input_ids, attention_mask)
    
    diff = (input_ids != aug_ids).sum().item()
    print(f"\nAugmentation: {diff} tokens modified out of {batch_size * seq_len}")
    
    print("\n=== Demo Complete ===")

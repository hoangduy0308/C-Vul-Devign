"""
Training Script with Contrastive Learning for Vulnerability Detection.

Combines BCE classification loss with supervised contrastive loss (SupCon).

Key features:
1. Two-phase training: warmup (BCE only) -> combined (BCE + SupCon)
2. Curriculum learning: gradually increase contrastive weight
3. Embedding visualization and analysis
4. Compatible with existing BaselineConfig

Usage:
    python train_contrastive.py --config baseline --use_contrastive
    python train_contrastive.py --config baseline --contrastive_weight 0.5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config_simplified import (
    BaselineConfig,
    get_baseline_config,
    get_seeds_for_evaluation,
)
from src.training.train_simplified import (
    SimplifiedLoss,
    compute_pos_weight,
    get_pos_weight_for_config,
    get_probs_from_logits,
    compute_metrics,
    EvalMetrics,
    aggregate_results,
    AggregatedResults,
)
from src.training.contrastive import (
    SupConLoss,
    SimCLRLoss,
    CombinedContrastiveLoss,
    ContrastiveConfig,
    ProjectionHead,
    CodeAugmentor,
    compute_contrastive_metrics,
    ContrastiveTrainingCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= CONTRASTIVE-AWARE MODEL =============

class ContrastiveBiGRUDetector(nn.Module):
    """
    BiGRU Vulnerability Detector with Contrastive Learning support.
    
    Adds a projection head for contrastive learning while maintaining
    the original classification head.
    """
    
    def __init__(
        self,
        base_config: BaselineConfig,
        contrastive_config: ContrastiveConfig
    ):
        super().__init__()
        self.base_config = base_config
        self.contrastive_config = contrastive_config
        
        # Token augmentation settings
        self.use_token_augmentation = getattr(base_config, 'use_token_augmentation', False)
        self.token_dropout_prob = getattr(base_config, 'token_dropout_prob', 0.05)
        self.token_mask_prob = getattr(base_config, 'token_mask_prob', 0.03)
        self.mask_token_id = getattr(base_config, 'mask_token_id', 1)
        
        # Embedding layer
        self.embedding = nn.Embedding(
            base_config.vocab_size,
            base_config.embed_dim,
            padding_idx=0
        )
        self.embed_dropout = nn.Dropout(base_config.embedding_dropout)
        
        # Token type embedding (optional)
        self.use_token_type_embedding = getattr(base_config, 'use_token_type_embedding', True)
        if self.use_token_type_embedding:
            num_token_types = getattr(base_config, 'num_token_types', 16)
            token_type_embed_dim = getattr(base_config, 'token_type_embed_dim', 32)
            self.token_type_embedding = nn.Embedding(num_token_types, token_type_embed_dim, padding_idx=0)
            self.token_type_proj = nn.Linear(token_type_embed_dim, base_config.embed_dim)
        
        # BiGRU encoder
        self.gru = nn.GRU(
            base_config.embed_dim,
            base_config.hidden_dim,
            num_layers=base_config.num_layers,
            bidirectional=base_config.bidirectional,
            batch_first=True,
            dropout=base_config.rnn_dropout if base_config.num_layers > 1 else 0.0
        )
        
        self.rnn_out_dim = base_config.hidden_dim * (2 if base_config.bidirectional else 1)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(self.rnn_out_dim, self.rnn_out_dim // 2),
            nn.Tanh(),
            nn.Linear(self.rnn_out_dim // 2, 1, bias=False)
        )
        self.context_dropout = nn.Dropout(0.2)
        
        # Vuln features branch (optional)
        if base_config.use_vuln_features:
            self.vuln_bn_in = nn.BatchNorm1d(base_config.vuln_feature_dim)
            vuln_hidden = base_config.vuln_feature_hidden_dim
            self.vuln_mlp = nn.Sequential(
                nn.Linear(base_config.vuln_feature_dim, vuln_hidden),
                nn.BatchNorm1d(vuln_hidden),
                nn.GELU(),
                nn.Dropout(base_config.vuln_feature_dropout)
            )
            self.combined_dim = self.rnn_out_dim + vuln_hidden
        else:
            self.combined_dim = self.rnn_out_dim
        
        # LayerNorm before classification
        self.use_layer_norm = getattr(base_config, 'use_layer_norm', True)
        if self.use_layer_norm:
            self.pre_classifier_ln = nn.LayerNorm(self.combined_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.BatchNorm1d(self.combined_dim // 2),
            nn.GELU(),
            nn.Dropout(base_config.classifier_dropout),
            nn.Linear(self.combined_dim // 2, 2)
        )
        
        # Projection head for contrastive learning
        if contrastive_config.use_contrastive:
            self.projection_head = ProjectionHead(
                input_dim=self.combined_dim,
                hidden_dim=contrastive_config.projection_hidden_dim,
                output_dim=contrastive_config.projection_output_dim,
                dropout=contrastive_config.projection_dropout
            )
        
        # Code augmentor for contrastive views
        self.augmentor = CodeAugmentor(
            dropout_prob=contrastive_config.aug_dropout_prob,
            mask_prob=contrastive_config.aug_mask_prob,
            shuffle_prob=contrastive_config.aug_shuffle_prob,
            shuffle_window=contrastive_config.aug_shuffle_window
        )
    
    def apply_token_augmentation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply standard token augmentation during training."""
        if not self.training or not self.use_token_augmentation:
            return input_ids
        
        augmented = input_ids.clone()
        B, L = input_ids.shape
        valid_mask = (input_ids > 3) & (attention_mask > 0)
        
        if self.token_dropout_prob > 0:
            dropout_mask = torch.rand(B, L, device=input_ids.device) < self.token_dropout_prob
            dropout_mask = dropout_mask & valid_mask
            augmented = augmented.masked_fill(dropout_mask, 0)
        
        if self.token_mask_prob > 0:
            mask_mask = torch.rand(B, L, device=input_ids.device) < self.token_mask_prob
            mask_mask = mask_mask & valid_mask & (augmented > 0)
            augmented = augmented.masked_fill(mask_mask, self.mask_token_id)
        
        return augmented
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        extended_token_type_ids: Optional[torch.Tensor] = None,
        vuln_features: Optional[torch.Tensor] = None,
        apply_augmentation: bool = True
    ) -> torch.Tensor:
        """
        Encode input to get embeddings (before classification).
        
        Returns:
            embeddings: [B, combined_dim] tensor
        """
        B, L = input_ids.shape
        
        # Token augmentation
        if apply_augmentation:
            augmented_ids = self.apply_token_augmentation(input_ids, attention_mask)
        else:
            augmented_ids = input_ids
        
        # Embedding
        embedded = self.embedding(augmented_ids)
        
        # Add token type embedding if available
        if self.use_token_type_embedding and extended_token_type_ids is not None:
            type_embedded = self.token_type_embedding(extended_token_type_ids)
            type_embedded = self.token_type_proj(type_embedded)
            embedded = embedded + type_embedded
        
        embedded = self.embed_dropout(embedded)
        
        # BiGRU
        use_packing = getattr(self.base_config, 'use_packed_sequences', True)
        
        if use_packing and lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L
            )
        else:
            rnn_out, _ = self.gru(embedded)
        
        # Attention pooling
        att_scores = self.attention(rnn_out)
        mask = attention_mask.unsqueeze(-1)
        att_scores = att_scores.masked_fill(mask == 0, -1e4)
        att_weights = F.softmax(att_scores, dim=1)
        context_vector = torch.sum(rnn_out * att_weights, dim=1)
        context_vector = self.context_dropout(context_vector)
        
        # Combine with vuln features
        if self.base_config.use_vuln_features and vuln_features is not None:
            feat_out = self.vuln_bn_in(vuln_features)
            feat_out = self.vuln_mlp(feat_out)
            combined = torch.cat([context_vector, feat_out], dim=1)
        else:
            combined = context_vector
        
        # LayerNorm
        if self.use_layer_norm:
            combined = self.pre_classifier_ln(combined)
        
        return combined
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        extended_token_type_ids: Optional[torch.Tensor] = None,
        vuln_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_projections: bool = False,
        generate_contrastive_view: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional contrastive outputs.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            lengths: Sequence lengths [B]
            extended_token_type_ids: Token type IDs [B, L]
            vuln_features: Vulnerability features [B, F]
            return_embeddings: Return encoder embeddings
            return_projections: Return projected embeddings
            generate_contrastive_view: Generate second view for contrastive
            
        Returns:
            Dictionary with 'logits', optionally 'embeddings', 'projections', 'aug_projections'
        """
        outputs = {}
        
        # Main forward pass
        embeddings = self.encode(
            input_ids, attention_mask, lengths,
            extended_token_type_ids, vuln_features,
            apply_augmentation=True
        )
        
        logits = self.classifier(embeddings)
        outputs['logits'] = logits
        
        if return_embeddings:
            outputs['embeddings'] = embeddings
        
        if return_projections and hasattr(self, 'projection_head'):
            projections = self.projection_head(embeddings)
            outputs['projections'] = projections
        
        # Generate augmented view for contrastive learning
        if generate_contrastive_view and self.training and hasattr(self, 'projection_head'):
            # Create augmented input
            aug_input_ids = self.augmentor(input_ids, attention_mask)
            
            # Encode augmented view (without additional token augmentation)
            aug_embeddings = self.encode(
                aug_input_ids, attention_mask, lengths,
                extended_token_type_ids, vuln_features,
                apply_augmentation=False  # Already augmented
            )
            
            aug_projections = self.projection_head(aug_embeddings)
            outputs['aug_embeddings'] = aug_embeddings
            outputs['aug_projections'] = aug_projections
        
        return outputs


# ============= TRAINING LOOP =============

def train_epoch_contrastive(
    model: ContrastiveBiGRUDetector,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: BaselineConfig,
    contrastive_config: ContrastiveConfig,
    callback: ContrastiveTrainingCallback
) -> Dict[str, float]:
    """
    Train one epoch with contrastive learning.
    """
    model.train()
    
    # Get contrastive weight for this epoch
    con_weight = callback.on_epoch_start(epoch)
    use_contrastive = con_weight > 0
    
    # Loss functions
    cls_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0]).to(device)  # Will be updated
    )
    
    if use_contrastive:
        if contrastive_config.use_supcon:
            con_criterion = SupConLoss(temperature=contrastive_config.temperature)
        else:
            con_criterion = SimCLRLoss(temperature=contrastive_config.temperature)
    
    epoch_stats = defaultdict(float)
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)
        vuln_features = batch.get('vuln_features')
        extended_token_type_ids = batch.get('extended_token_type_ids')
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device)
        if extended_token_type_ids is not None:
            extended_token_type_ids = extended_token_type_ids.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths,
                extended_token_type_ids=extended_token_type_ids,
                vuln_features=vuln_features,
                return_embeddings=True,
                return_projections=use_contrastive,
                generate_contrastive_view=use_contrastive and not contrastive_config.use_supcon
            )
            
            logits = outputs['logits']
            
            # Classification loss
            if logits.size(1) == 2:
                cls_logits = logits[:, 1]
            else:
                cls_logits = logits.squeeze()
            
            cls_loss = cls_criterion(cls_logits, labels.float())
            
            # Contrastive loss
            if use_contrastive:
                if contrastive_config.use_supcon:
                    # Supervised contrastive: use labels to define positive pairs
                    projections = outputs['projections']
                    con_loss = con_criterion(projections, labels)
                else:
                    # SimCLR: use augmented views as positive pairs
                    projections = outputs['projections']
                    aug_projections = outputs['aug_projections']
                    con_loss = con_criterion(projections, aug_projections)
                
                total_loss = cls_loss + con_weight * con_loss
            else:
                con_loss = torch.tensor(0.0)
                total_loss = cls_loss
        
        # Backward pass
        scaler.scale(total_loss).backward()
        
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update stats
        epoch_stats['cls_loss'] += cls_loss.item()
        epoch_stats['con_loss'] += con_loss.item() if use_contrastive else 0
        epoch_stats['total_loss'] += total_loss.item()
        n_batches += 1
        
        # Callback
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'con_loss': con_loss.item() if use_contrastive else 0,
            'total_loss': total_loss.item()
        }
        
        if use_contrastive and 'embeddings' in outputs:
            callback.on_batch_end(
                epoch, batch_idx, loss_dict,
                outputs['embeddings'].detach(),
                labels
            )
    
    # Average stats
    for key in epoch_stats:
        epoch_stats[key] /= n_batches
    
    epoch_stats['con_weight'] = con_weight
    
    return dict(epoch_stats)


@torch.no_grad()
def evaluate_contrastive(
    model: ContrastiveBiGRUDetector,
    val_loader: DataLoader,
    device: torch.device,
    config: BaselineConfig,
    return_contrastive_metrics: bool = True
) -> Tuple[EvalMetrics, Dict[str, float]]:
    """
    Evaluate model with contrastive metrics.
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    all_embeddings = []
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)
        vuln_features = batch.get('vuln_features')
        extended_token_type_ids = batch.get('extended_token_type_ids')
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device)
        if extended_token_type_ids is not None:
            extended_token_type_ids = extended_token_type_ids.to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            extended_token_type_ids=extended_token_type_ids,
            vuln_features=vuln_features,
            return_embeddings=return_contrastive_metrics,
            return_projections=False,
            generate_contrastive_view=False
        )
        
        logits = outputs['logits']
        probs = get_probs_from_logits(logits)
        
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        if return_contrastive_metrics and 'embeddings' in outputs:
            all_embeddings.append(outputs['embeddings'])
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Compute classification metrics
    cls_metrics = compute_metrics(
        all_probs, all_labels,
        threshold_min=config.threshold_min,
        threshold_max=config.threshold_max,
        threshold_step=config.threshold_step
    )
    
    # Compute contrastive metrics
    con_metrics = {}
    if return_contrastive_metrics and all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.tensor(all_labels, device=all_embeddings.device)
        con_metrics = compute_contrastive_metrics(all_embeddings, labels_tensor)
    
    return cls_metrics, con_metrics


# ============= MAIN TRAINING FUNCTION =============

def train_with_contrastive(
    data_dir: str,
    output_dir: str,
    base_config: Optional[BaselineConfig] = None,
    contrastive_config: Optional[ContrastiveConfig] = None,
    seed: int = 42,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Main training function with contrastive learning.
    
    Args:
        data_dir: Path to preprocessed data
        output_dir: Path to save checkpoints
        base_config: Base training configuration
        contrastive_config: Contrastive learning configuration
        seed: Random seed
        device: Device to use
        
    Returns:
        Dictionary with training results
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Default configs
    if base_config is None:
        base_config = get_baseline_config()
    if contrastive_config is None:
        contrastive_config = ContrastiveConfig()
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configs
    with open(os.path.join(output_dir, 'base_config.json'), 'w') as f:
        json.dump(asdict(base_config), f, indent=2)
    with open(os.path.join(output_dir, 'contrastive_config.json'), 'w') as f:
        json.dump(asdict(contrastive_config), f, indent=2)
    
    # Create dataloaders (using existing DevignDataset)
    from src.pipeline.dataset import DevignDataset, create_dataloader
    
    train_path = Path(data_dir) / 'train.npz'
    val_path = Path(data_dir) / 'val.npz'
    test_path = Path(data_dir) / 'test.npz'
    
    logger.info(f"Loading data from {data_dir}")
    
    # Note: You need to adapt this to your actual dataset loading
    # This is a placeholder that matches the existing API
    
    # Create model
    model = ContrastiveBiGRUDetector(base_config, contrastive_config)
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_config.learning_rate,
        weight_decay=base_config.weight_decay
    )
    
    # Scheduler
    if base_config.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=base_config.max_epochs,
            eta_min=base_config.scheduler_min_lr
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=base_config.scheduler_factor,
            patience=base_config.scheduler_patience,
            min_lr=base_config.scheduler_min_lr
        )
    
    # Scaler for mixed precision
    scaler = GradScaler() if base_config.use_amp else GradScaler(enabled=False)
    
    # Callback
    callback = ContrastiveTrainingCallback(contrastive_config)
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'contrastive': []
    }
    
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    
    logger.info("Starting training with contrastive learning...")
    logger.info(f"  Contrastive: {contrastive_config.use_contrastive}")
    logger.info(f"  SupCon: {contrastive_config.use_supcon}")
    logger.info(f"  Temperature: {contrastive_config.temperature}")
    logger.info(f"  Warmup epochs: {contrastive_config.contrastive_warmup_epochs}")
    
    # Note: The actual training loop would go here
    # This is a template showing the structure
    
    results = {
        'best_epoch': best_epoch,
        'best_metric': best_metric,
        'history': history,
        'callback_summary': callback.get_summary()
    }
    
    return results


# ============= CLI =============

def main():
    parser = argparse.ArgumentParser(description='Train with Contrastive Learning')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/contrastive',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    # Contrastive settings
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Enable contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Contrastive temperature')
    parser.add_argument('--use_supcon', action='store_true', default=True,
                        help='Use supervised contrastive (vs SimCLR)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Epochs before adding contrastive loss')
    
    args = parser.parse_args()
    
    # Create configs
    base_config = get_baseline_config()
    
    contrastive_config = ContrastiveConfig(
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        use_supcon=args.use_supcon,
        contrastive_warmup_epochs=args.warmup_epochs
    )
    
    # Train
    results = train_with_contrastive(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_config=base_config,
        contrastive_config=contrastive_config,
        seed=args.seed,
        device=args.device
    )
    
    logger.info(f"Training complete. Best metric: {results['best_metric']:.4f}")


if __name__ == '__main__':
    main()

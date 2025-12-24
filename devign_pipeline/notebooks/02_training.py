# %% [markdown]
# # Devign Model Training Pipeline V2 - Simplified
# 
# Clean, maintainable training script using:
# - **BaselineConfig** from config_simplified.py (replaces 20+ config classes)
# - **SimplifiedLoss** (BCEWithLogitsLoss with pos_weight)
# - **Multi-seed evaluation** with mean ± std reporting
# - **SWA support** for improved generalization
# 
# Model: ImprovedHybridBiGRUVulnDetector (unchanged from v1)

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import sys
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from tqdm.auto import tqdm

# Environment setup
if os.path.exists('/kaggle/input'):
    WORKING_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input/devign-final/processed'
    sys.path.insert(0, '/tmp/devign_pipeline')
else:
    WORKING_DIR = 'f:/Work/C Vul Devign'
    DATA_DIR = 'f:/Work/C Vul Devign/Dataset/devign_final'
    sys.path.insert(0, 'f:/Work/C Vul Devign/devign_pipeline')

# Import simplified config and loss
from src.training.config_simplified import (
    BaselineConfig,
    get_baseline_config,
    get_recall_focused_config,
    get_precision_focused_config,
    get_large_config,
    get_seeds_for_evaluation,
)
from src.training.train_simplified import (
    SimplifiedLoss,
    compute_pos_weight,
    get_pos_weight_for_config,
    compute_metrics,
    EvalMetrics,
    aggregate_results,
    AggregatedResults,
)

MODEL_DIR = os.path.join(WORKING_DIR, 'models')
LOG_DIR = os.path.join(WORKING_DIR, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    gpu_info = ', '.join([f"{torch.cuda.get_device_properties(i).name}" for i in range(N_GPUS)])
    print(f"Device: {DEVICE} ({N_GPUS}x {gpu_info})")
else:
    print(f"Device: {DEVICE}")


# %% [markdown]
# ## 2. Dataset

# %%
class DevignDataset(Dataset):
    """Load preprocessed .npz file with segment-aware features."""
    
    SEP_TOKEN_ID = 4
    
    def __init__(
        self, 
        npz_path: str, 
        max_seq_length: int = 512, 
        load_vuln_features: bool = False, 
        vuln_feature_dim: int = 25
    ):
        self.max_seq_length = max_seq_length
        self.load_vuln_features = load_vuln_features
        self.vuln_feature_dim = vuln_feature_dim
        
        data = np.load(npz_path)
        self.input_ids = data['input_ids']
        self.labels = data['labels']
        self.attention_mask = data.get('attention_mask', None)
        
        self.vuln_features = None
        if self.load_vuln_features:
            path_obj = Path(npz_path)
            vuln_path = path_obj.with_name(f"{path_obj.stem}_vuln.npz")
            if vuln_path.exists():
                vuln_data = np.load(vuln_path)
                self.vuln_features = vuln_data.get('features') or vuln_data.get('vuln_features')
                if self.vuln_features is not None and len(self.vuln_features) != len(self.labels):
                    self.vuln_features = None
    
    def _find_sep_position(self, input_ids: np.ndarray) -> int:
        sep_positions = np.where(input_ids == self.SEP_TOKEN_ID)[0]
        if len(sep_positions) > 0:
            return int(sep_positions[0])
        non_pad = np.sum(input_ids != 0)
        return int(non_pad * 0.8)
    
    def _create_token_type_ids(self, input_ids: np.ndarray, sep_pos: int) -> np.ndarray:
        token_type_ids = np.zeros(len(input_ids), dtype=np.int64)
        token_type_ids[sep_pos + 1:] = 1
        return token_type_ids
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_ids = self.input_ids[idx]
        input_ids = raw_ids[:self.max_seq_length]
        
        sep_pos = self._find_sep_position(input_ids)
        sep_pos = min(sep_pos, self.max_seq_length - 1)
        token_type_ids = self._create_token_type_ids(input_ids, sep_pos)
        
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = np.pad(input_ids, (0, padding_length), constant_values=0)
            token_type_ids = np.pad(token_type_ids, (0, padding_length), constant_values=0)
        
        if self.attention_mask is not None:
            attention_mask = self.attention_mask[idx][:self.max_seq_length]
            if len(attention_mask) < self.max_seq_length:
                attention_mask = np.pad(attention_mask, (0, self.max_seq_length - len(attention_mask)), constant_values=0)
            orig_len = int(np.sum(attention_mask[:self.max_seq_length]))
        else:
            attention_mask = (input_ids != 0).astype(np.float32)
            orig_len = int(np.sum(attention_mask))
        
        orig_len = max(1, min(orig_len, self.max_seq_length))
        
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'sep_pos': torch.tensor(sep_pos, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'lengths': torch.tensor(orig_len, dtype=torch.long)
        }
        
        if self.load_vuln_features:
            if self.vuln_features is not None:
                item['vuln_features'] = torch.tensor(self.vuln_features[idx], dtype=torch.float)
            else:
                item['vuln_features'] = torch.zeros(self.vuln_feature_dim, dtype=torch.float)
        
        return item


def create_dataloaders(
    data_dir: str, 
    batch_size: int, 
    max_seq_length: int,
    num_workers: int = 2,
    load_vuln_features: bool = False,
    vuln_feature_dim: int = 25
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    train_path = Path(data_dir) / 'train.npz'
    val_path = Path(data_dir) / 'val.npz'
    test_path = Path(data_dir) / 'test.npz'
    
    if not train_path.exists():
        raise ValueError(f"Could not find train.npz in {data_dir}")
    
    train_dataset = DevignDataset(str(train_path), max_seq_length, load_vuln_features, vuln_feature_dim)
    val_dataset = DevignDataset(str(val_path), max_seq_length, load_vuln_features, vuln_feature_dim)
    test_dataset = DevignDataset(str(test_path), max_seq_length, load_vuln_features, vuln_feature_dim)
    
    print(f"Data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    loader_kwargs = dict(
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


# %% [markdown]
# ## 3. Model Architecture (ImprovedHybridBiGRUVulnDetector)

# %%
class MultiHeadSelfAttentionPooling(nn.Module):
    """Multi-head self-attention pooling over BiGRU outputs."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, rnn_outputs: torch.Tensor, attention_mask: torch.Tensor):
        bsz = rnn_outputs.size(0)
        query = self.query.expand(bsz, -1, -1)
        key_padding_mask = ~attention_mask.bool()
        
        attn_output, _ = self.mha(
            query, rnn_outputs, rnn_outputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        context = attn_output.squeeze(1)
        return self.dropout(context)


class ImprovedHybridBiGRUVulnDetector(nn.Module):
    """
    Improved Hybrid BiGRU with:
    1. Packed sequences support
    2. Multi-head attention pooling
    3. Token augmentation during training
    4. LayerNorm before classifier
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        self.use_token_augmentation = getattr(config, 'use_token_augmentation', False)
        self.token_dropout_prob = getattr(config, 'token_dropout_prob', 0.05)
        self.token_mask_prob = getattr(config, 'token_mask_prob', 0.03)
        self.mask_token_id = getattr(config, 'mask_token_id', 1)
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        # BiGRU
        self.gru = nn.GRU(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.rnn_dropout if config.num_layers > 1 else 0.0
        )
        
        self.rnn_out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Attention
        self.use_multihead_attention = getattr(config, 'use_multihead_attention', True)
        if self.use_multihead_attention:
            self.attention = MultiHeadSelfAttentionPooling(
                input_dim=self.rnn_out_dim,
                num_heads=getattr(config, 'num_attention_heads', 4),
                dropout=getattr(config, 'attention_dropout', 0.1),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.rnn_out_dim, self.rnn_out_dim // 2),
                nn.Tanh(),
                nn.Linear(self.rnn_out_dim // 2, 1, bias=False)
            )
        self.context_dropout = nn.Dropout(0.2)
        
        # Vuln features branch
        if config.use_vuln_features:
            self.vuln_bn_in = nn.BatchNorm1d(config.vuln_feature_dim)
            vuln_hidden = config.vuln_feature_hidden_dim
            self.vuln_mlp = nn.Sequential(
                nn.Linear(config.vuln_feature_dim, vuln_hidden),
                nn.BatchNorm1d(vuln_hidden),
                nn.GELU(),
                nn.Dropout(config.vuln_feature_dropout)
            )
            self.combined_dim = self.rnn_out_dim + vuln_hidden
        else:
            self.combined_dim = self.rnn_out_dim
        
        # LayerNorm
        self.use_layer_norm = getattr(config, 'use_layer_norm', True)
        if self.use_layer_norm:
            self.pre_classifier_ln = nn.LayerNorm(self.combined_dim)
        
        # Classifier (outputs 2 classes for compatibility, but we use class 1 logit for BCE)
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.BatchNorm1d(self.combined_dim // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.combined_dim // 2, 2)
        )
    
    def apply_token_augmentation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
    
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        vuln_features=None, 
        lengths=None,
        token_type_ids=None, 
        sep_pos=None
    ):
        B, L = input_ids.shape
        
        augmented_ids = self.apply_token_augmentation(input_ids, attention_mask)
        embedded = self.embedding(augmented_ids)
        embedded = self.embed_dropout(embedded)
        
        use_packing = getattr(self.config, 'use_packed_sequences', True)
        
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
        
        if self.use_multihead_attention:
            context_vector = self.attention(rnn_out, attention_mask)
        else:
            att_scores = self.attention(rnn_out)
            mask = attention_mask.unsqueeze(-1)
            att_scores = att_scores.masked_fill(mask == 0, -1e4)
            att_weights = F.softmax(att_scores, dim=1)
            context_vector = torch.sum(rnn_out * att_weights, dim=1)
            context_vector = self.context_dropout(context_vector)
        
        if self.config.use_vuln_features and vuln_features is not None:
            feat_out = self.vuln_bn_in(vuln_features)
            feat_out = self.vuln_mlp(feat_out)
            combined = torch.cat([context_vector, feat_out], dim=1)
        else:
            combined = context_vector
        
        if self.use_layer_norm:
            combined = self.pre_classifier_ln(combined)
        
        logits = self.classifier(combined)
        return logits


def load_pretrained_embedding(config: BaselineConfig, data_dir: str) -> Optional[np.ndarray]:
    """Load pretrained embedding matrix if available."""
    if not config.use_pretrained_embedding:
        return None
    
    emb_path = config.embedding_path
    if not emb_path:
        possible_paths = [
            os.path.join(data_dir, 'embedding_matrix.npy'),
            '/kaggle/working/embedding_matrix.npy',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                emb_path = p
                break
    
    if emb_path and os.path.exists(emb_path):
        pretrained = np.load(emb_path)
        print(f"Loaded pretrained embedding: {pretrained.shape}")
        return pretrained
    else:
        print("Pretrained embedding not found, using random init")
        return None


def build_model(config: BaselineConfig, pretrained_embedding: Optional[np.ndarray] = None) -> nn.Module:
    """Build and initialize model."""
    model = ImprovedHybridBiGRUVulnDetector(config)
    
    if pretrained_embedding is not None:
        model.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        if config.freeze_embedding:
            model.embedding.weight.requires_grad = False
            print("Pretrained embedding FROZEN")
        else:
            print(f"Pretrained embedding TRAINABLE (LR scale={config.embedding_lr_scale})")
    
    model.to(DEVICE)
    if N_GPUS > 1:
        model = nn.DataParallel(model)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {params:,} params, hidden={config.hidden_dim}, layers={config.num_layers}")
    
    return model


# %% [markdown]
# ## 4. Training Utilities

# %%
class EarlyStopping:
    """Early stopping with maximize/minimize modes."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'maximize'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'maximize':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def find_optimal_threshold(
    labels: np.ndarray, 
    probs: np.ndarray,
    min_t: float = 0.3,
    max_t: float = 0.7,
    step: float = 0.01
) -> Tuple[float, float]:
    """Find threshold that maximizes F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(min_t, max_t + step, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


@torch.no_grad()
def update_bn_for_swa(loader: DataLoader, model: nn.Module, device: torch.device):
    """Update BatchNorm statistics for SWA model."""
    from torch.nn.modules.batchnorm import _BatchNorm
    
    has_bn = any(isinstance(m, _BatchNorm) for m in model.modules())
    if not has_bn:
        return
    
    was_training = model.training
    model.train()
    
    momenta = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
    
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        vuln_features = batch.get("vuln_features")
        lengths = batch.get("lengths")
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device, non_blocking=True)
        if lengths is not None:
            lengths = lengths.to(device, non_blocking=True)
        
        b = input_ids.size(0)
        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum
        
        model(input_ids, attention_mask, vuln_features, lengths)
        n += b
    
    for module, mom in momenta.items():
        module.momentum = mom
    
    model.train(was_training)


# %% [markdown]
# ## 5. Training Loop

# %%
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    config: BaselineConfig,
    device: torch.device
) -> Dict[str, float]:
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        vuln_features = batch.get('vuln_features')
        lengths = batch['lengths'].to(device, non_blocking=True)
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            logits = model(input_ids, attention_mask, vuln_features, lengths)
            loss = criterion(logits, labels)
            loss = loss / config.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.accumulation_steps
        
        probs = torch.sigmoid(logits.detach()[:, 1]).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': avg_loss,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': auc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: BaselineConfig,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model with optimal threshold search."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        vuln_features = batch.get('vuln_features')
        lengths = batch['lengths'].to(device, non_blocking=True)
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            logits = model(input_ids, attention_mask, vuln_features, lengths)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits[:, 1]).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    best_t, best_f1 = find_optimal_threshold(
        all_labels, all_probs,
        min_t=config.threshold_min,
        max_t=config.threshold_max,
        step=config.threshold_step
    )
    
    preds = (all_probs >= best_t).astype(int)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': avg_loss,
        'f1': best_f1,
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'auc': auc,
        'threshold': best_t,
        'probs': all_probs,
        'labels': all_labels,
    }


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% [markdown]
# ## 6. Single Seed Training

# %%
def train_single_seed(
    config: BaselineConfig,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pretrained_embedding: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[EvalMetrics, nn.Module]:
    """
    Train model with a single seed.
    
    Returns:
        metrics: EvalMetrics from best checkpoint
        model: Trained model (SWA if enabled, else best checkpoint)
    """
    set_seed(seed)
    
    # Build model
    model = build_model(config, pretrained_embedding)
    
    # Compute pos_weight from training data or use override
    train_labels = train_loader.dataset.labels
    pos_weight = get_pos_weight_for_config(config, train_labels)
    
    # Create loss function
    criterion = SimplifiedLoss(
        loss_type=config.loss_type,
        pos_weight=pos_weight,
        label_smoothing=config.label_smoothing,
    ).to(DEVICE)
    
    # Optimizer with differential LR for pretrained embeddings
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    if config.use_pretrained_embedding and not config.freeze_embedding and config.embedding_lr_scale != 1.0:
        embedding_params = list(base_model.embedding.parameters())
        embedding_ids = {id(p) for p in embedding_params}
        other_params = [p for p in model.parameters() if id(p) not in embedding_ids and p.requires_grad]
        
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': config.learning_rate},
            {'params': embedding_params, 'lr': config.learning_rate * config.embedding_lr_scale}
        ], weight_decay=config.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Scheduler
    if config.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=config.scheduler_min_lr
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.scheduler_factor,
            patience=config.scheduler_patience, min_lr=config.scheduler_min_lr
        )
    
    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # SWA setup
    swa_model = None
    swa_scheduler = None
    if config.use_swa:
        swa_model = AveragedModel(base_model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr, anneal_epochs=2)
    
    # Training loop
    best_f1, best_epoch = 0.0, 0
    best_state = None
    best_metrics = None
    
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, config, DEVICE)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, config, DEVICE)
        
        epoch_time = time.time() - epoch_start
        
        if verbose:
            print(
                f"  Ep {epoch:2d}/{config.max_epochs} ({epoch_time:.0f}s) | "
                f"Train: L={train_metrics['loss']:.3f} F1={train_metrics['f1']:.3f} | "
                f"Val: F1={val_metrics['f1']:.3f} AUC={val_metrics['auc']:.3f} T={val_metrics['threshold']:.2f}"
            )
        
        # Scheduler step
        if config.use_swa and epoch >= config.swa_start_epoch:
            swa_model.update_parameters(base_model)
            swa_scheduler.step()
        elif config.scheduler_type == 'plateau':
            scheduler.step(val_metrics['auc'])
        else:
            scheduler.step()
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
            best_metrics = val_metrics
        
        if early_stopping(val_metrics['f1']):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break
    
    # SWA finalization
    if config.use_swa and swa_model is not None:
        update_bn_for_swa(train_loader, swa_model, DEVICE)
        swa_metrics = evaluate(swa_model, val_loader, criterion, config, DEVICE)
        
        if verbose:
            print(f"  [SWA] Val: F1={swa_metrics['f1']:.4f} AUC={swa_metrics['auc']:.4f}")
        
        if swa_metrics['f1'] > best_f1:
            best_metrics = swa_metrics
            final_model = swa_model
            if verbose:
                print(f"  [SWA] Using SWA model (F1={swa_metrics['f1']:.4f} > {best_f1:.4f})")
        else:
            base_model.load_state_dict(best_state)
            final_model = model
    else:
        base_model.load_state_dict(best_state)
        final_model = model
    
    eval_metrics = EvalMetrics(
        f1=best_metrics['f1'],
        precision=best_metrics['precision'],
        recall=best_metrics['recall'],
        auc=best_metrics['auc'],
        accuracy=accuracy_score(best_metrics['labels'], (best_metrics['probs'] >= best_metrics['threshold']).astype(int)),
        threshold=best_metrics['threshold'],
    )
    
    return eval_metrics, final_model


# %% [markdown]
# ## 7. Multi-Seed Training

# %%
def train_multi_seed(
    config: BaselineConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    pretrained_embedding: Optional[np.ndarray] = None,
    n_seeds: int = 3,
) -> AggregatedResults:
    """
    Train with multiple seeds and report mean ± std.
    
    Oracle recommendation: Report mean ± std across 3+ seeds.
    """
    seeds = get_seeds_for_evaluation(n_seeds, config.ensemble_base_seed)
    
    print(f"\n{'='*60}")
    print(f"MULTI-SEED TRAINING: {n_seeds} seeds")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}\n")
    
    all_results = []
    all_models = []
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        metrics, model = train_single_seed(
            config, seed, train_loader, val_loader, pretrained_embedding
        )
        all_results.append(metrics)
        all_models.append(model)
        
        print(f"  → F1={metrics.f1:.4f}, AUC={metrics.auc:.4f}, T={metrics.threshold:.2f}")
    
    # Aggregate results
    aggregated = aggregate_results(all_results, config_name="multi_seed")
    
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(aggregated)
    
    # Test set evaluation with best model (highest val F1)
    best_idx = np.argmax([r.f1 for r in all_results])
    best_model = all_models[best_idx]
    
    print(f"\n--- Test Set Evaluation (using seed {seeds[best_idx]}) ---")
    
    # Create criterion for test eval
    test_labels = test_loader.dataset.labels
    pos_weight = get_pos_weight_for_config(config, test_labels)
    criterion = SimplifiedLoss(loss_type=config.loss_type, pos_weight=pos_weight).to(DEVICE)
    
    test_metrics = evaluate(best_model, test_loader, criterion, config, DEVICE)
    print(f"Test: F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} "
          f"R={test_metrics['recall']:.4f} AUC={test_metrics['auc']:.4f}")
    
    return aggregated


# %% [markdown]
# ## 8. Main Training Script

# %%
def main(
    config_name: str = "baseline",
    n_seeds: int = 3,
    data_dir: str = None,
):
    """
    Main training function.
    
    Args:
        config_name: One of "baseline", "recall_focused", "precision_focused", "large"
        n_seeds: Number of seeds for multi-seed evaluation
        data_dir: Path to preprocessed data
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Get config
    config_getters = {
        "baseline": get_baseline_config,
        "recall_focused": get_recall_focused_config,
        "precision_focused": get_precision_focused_config,
        "large": get_large_config,
    }
    
    if config_name not in config_getters:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(config_getters.keys())}")
    
    config = config_getters[config_name]()
    
    print(f"{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*60}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  loss_type: {config.loss_type}")
    print(f"  use_swa: {config.use_swa}")
    print(f"  n_seeds: {n_seeds}")
    print(f"{'='*60}\n")
    
    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        config.batch_size,
        config.max_seq_length,
        config.num_workers,
        load_vuln_features=config.use_vuln_features,
        vuln_feature_dim=config.vuln_feature_dim
    )
    
    # Print class distribution
    train_labels = train_loader.dataset.labels
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    print(f"Class distribution: {n_neg} neg, {n_pos} pos (ratio: {n_neg/n_pos:.2f}:1)")
    
    # Load pretrained embedding
    pretrained_embedding = load_pretrained_embedding(config, data_dir)
    
    # Train
    if n_seeds == 1:
        # Single seed training
        metrics, model = train_single_seed(
            config, 
            config.ensemble_base_seed, 
            train_loader, 
            val_loader, 
            pretrained_embedding
        )
        print(f"\nFinal: F1={metrics.f1:.4f}, AUC={metrics.auc:.4f}")
        
        # Save model
        save_path = os.path.join(MODEL_DIR, 'best_model_v2.pt')
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'model_state_dict': base_model.state_dict(),
            'config': config.to_dict(),
            'metrics': metrics.to_dict(),
        }, save_path)
        print(f"Saved model to {save_path}")
        
    else:
        # Multi-seed training
        results = train_multi_seed(
            config,
            train_loader,
            val_loader,
            test_loader,
            pretrained_embedding,
            n_seeds=n_seeds
        )
        
        # Save results
        results_path = os.path.join(LOG_DIR, f'{config_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'config': config.to_dict(),
                'n_seeds': n_seeds,
                'f1_mean': results.f1_mean,
                'f1_std': results.f1_std,
                'precision_mean': results.precision_mean,
                'recall_mean': results.recall_mean,
                'auc_mean': results.auc_mean,
            }, f, indent=2)
        print(f"\nSaved results to {results_path}")


# %% [markdown]
# ## 9. Run Training

# %%
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Devign Training V2")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=["baseline", "recall_focused", "precision_focused", "large"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    main(
        config_name=args.config,
        n_seeds=args.seeds,
        data_dir=args.data_dir,
    )

# %%
# Quick run for notebook:
# Uncomment the line below to run training
# main(config_name="baseline", n_seeds=3)

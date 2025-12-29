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
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModelOutput(NamedTuple):
    """Structured output from ImprovedHybridBiGRUVulnDetector forward pass."""
    logits: torch.Tensor
    embeddings: Optional[torch.Tensor] = None
    projections: Optional[torch.Tensor] = None


from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    get_balanced_pr_config,
    get_large_config,
    get_focal_config,
    get_improved_config,
    get_conservative_config,
    get_no_pretrained_config,
    get_seeds_for_evaluation,
    get_contrastive_config,
    get_simclr_config,
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

MODEL_DIR = os.path.join(WORKING_DIR, 'models')
LOG_DIR = os.path.join(WORKING_DIR, 'logs')
PLOT_DIR = os.path.join(WORKING_DIR, 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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
    """Load preprocessed .npz file with segment-aware features and token type IDs."""
    
    SEP_TOKEN_ID = 4
    
    def __init__(
        self, 
        npz_path: str, 
        max_seq_length: int = 512, 
        load_vuln_features: bool = False, 
        vuln_feature_dim: Optional[int] = None  # Uses BaselineConfig.vuln_feature_dim if None
    ):
        if vuln_feature_dim is None:
            vuln_feature_dim = BaselineConfig().vuln_feature_dim
        self.max_seq_length = max_seq_length
        self.load_vuln_features = load_vuln_features
        self.vuln_feature_dim = vuln_feature_dim
        
        data = np.load(npz_path)
        self.input_ids = data['input_ids']
        self.labels = data['labels']
        self.attention_mask = data.get('attention_mask', None)
        
        # Load token_type_ids if available (for extended token type embeddings)
        self.extended_token_type_ids = data.get('token_type_ids', None)
        if self.extended_token_type_ids is not None:
            print(f"  Loaded token_type_ids: {self.extended_token_type_ids.shape}")
        
        self.vuln_features = None
        if self.load_vuln_features:
            path_obj = Path(npz_path)
            vuln_path = path_obj.with_name(f"{path_obj.stem}_vuln.npz")
            if vuln_path.exists():
                vuln_data = np.load(vuln_path)
                # Check both possible key names
                if 'features' in vuln_data:
                    self.vuln_features = vuln_data['features']
                elif 'vuln_features' in vuln_data:
                    self.vuln_features = vuln_data['vuln_features']
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
        
        # Add extended token type IDs if available (for vulnerability-relevant type embedding)
        if self.extended_token_type_ids is not None:
            ext_type_ids = self.extended_token_type_ids[idx][:self.max_seq_length]
            if len(ext_type_ids) < self.max_seq_length:
                ext_type_ids = np.pad(ext_type_ids, (0, self.max_seq_length - len(ext_type_ids)), constant_values=0)
            item['extended_token_type_ids'] = torch.tensor(ext_type_ids, dtype=torch.long)
        
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
    vuln_feature_dim: Optional[int] = None  # Uses BaselineConfig.vuln_feature_dim if None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    if vuln_feature_dim is None:
        vuln_feature_dim = BaselineConfig().vuln_feature_dim
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
    5. Token type embedding for vulnerability-relevant semantic types
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        
        self.use_token_augmentation = getattr(config, 'use_token_augmentation', False)
        self.token_dropout_prob = getattr(config, 'token_dropout_prob', 0.05)
        self.token_mask_prob = getattr(config, 'token_mask_prob', 0.03)
        self.mask_token_id = getattr(config, 'mask_token_id', 1)
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        # Token type embedding (for vulnerability-relevant types)
        self.use_token_type_embedding = getattr(config, 'use_token_type_embedding', True)
        if self.use_token_type_embedding:
            num_token_types = getattr(config, 'num_token_types', 16)
            token_type_embed_dim = getattr(config, 'token_type_embed_dim', 32)
            self.token_type_embedding = nn.Embedding(num_token_types, token_type_embed_dim, padding_idx=0)
            # Project token type embedding to match embed_dim for additive combination
            self.token_type_proj = nn.Linear(token_type_embed_dim, config.embed_dim)
            print(f"  Token type embedding: {num_token_types} types -> {token_type_embed_dim}d -> {config.embed_dim}d")
        
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
        
        # Projection head for contrastive learning
        self.use_contrastive = getattr(config, 'use_contrastive', False)
        if self.use_contrastive:
            projection_hidden = getattr(config, 'projection_hidden_dim', 256)
            projection_output = getattr(config, 'projection_output_dim', 128)
            self.projection_head = ProjectionHead(
                input_dim=self.combined_dim,
                hidden_dim=projection_hidden,
                output_dim=projection_output,
                dropout=0.1
            )
            # Augmentor for generating contrastive views
            # NOTE: shuffle_prob=0 vì shuffling sẽ làm sai lệch extended_token_type_ids
            # Token type IDs gắn liền với vị trí gốc, nếu shuffle tokens mà không shuffle
            # type IDs thì embedding sẽ bị sai ngữ nghĩa
            self.contrastive_augmentor = CodeAugmentor(
                dropout_prob=0.20,  # Increased from 0.15
                mask_prob=0.15,     # Increased from 0.10
                shuffle_prob=0.0   # Keep disabled for token_type_ids alignment
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
        sep_pos=None,
        extended_token_type_ids=None,
        return_embeddings=False,
        return_projections=False,
    ):
        B, L = input_ids.shape
        
        augmented_ids = self.apply_token_augmentation(input_ids, attention_mask)
        embedded = self.embedding(augmented_ids)
        
        # Add token type embedding if available and enabled
        if self.use_token_type_embedding and extended_token_type_ids is not None:
            type_embedded = self.token_type_embedding(extended_token_type_ids)
            type_embedded = self.token_type_proj(type_embedded)
            embedded = embedded + type_embedded
        
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
        
        embeddings = combined if return_embeddings else None
        projections = None
        
        if return_projections and self.use_contrastive:
            projections = self.projection_head(combined)
        
        return ModelOutput(
            logits=logits,
            embeddings=embeddings,
            projections=projections,
        )
    
    def get_augmented_input(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate augmented input_ids for contrastive learning.
        
        Call this BEFORE forward() for the augmented view to save VRAM.
        """
        if self.use_contrastive and self.contrastive_augmentor is not None:
            return self.contrastive_augmentor(input_ids, attention_mask)
        return input_ids


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    
    Oracle recommendation: Post-hoc calibration helps raise precision 
    at same recall when logits are overconfident.
    
    Usage:
        1. Train model normally
        2. Fit temperature on validation set
        3. Apply temperature scaling during inference
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(
        self, 
        model: nn.Module, 
        val_loader: DataLoader, 
        config: BaselineConfig,
        device: torch.device,
        max_iter: int = 50
    ):
        """
        Fit temperature parameter on validation set using NLL loss.
        
        Args:
            model: Trained model
            val_loader: Validation DataLoader
            config: Training config
            device: Device to use
            max_iter: Maximum optimization iterations
        """
        model.eval()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                vuln_features = batch.get('vuln_features')
                lengths = batch['lengths'].to(device)
                extended_token_type_ids = batch.get('extended_token_type_ids')
                
                if vuln_features is not None:
                    vuln_features = vuln_features.to(device)
                if extended_token_type_ids is not None:
                    extended_token_type_ids = extended_token_type_ids.to(device)
                
                outputs = model(input_ids, attention_mask, vuln_features, lengths, 
                              extended_token_type_ids=extended_token_type_ids)
                
                all_logits.append(outputs.logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        
        print(f"  Temperature scaling: T = {self.temperature.item():.4f}")
        return self.temperature.item()


def load_pretrained_embedding(config: BaselineConfig, data_dir: str) -> Optional[np.ndarray]:
    """Load pretrained embedding matrix if available.
    
    Note: Word2Vec has been removed from the pipeline. This function now
    always returns None, using random initialization for embeddings.
    """
    return None


def load_vocab_size_from_data(data_dir: str, tokenizer_type: Optional[str] = None) -> int:
    """Load vocab_size from config.json or vocab.json in data directory.
    
    Args:
        data_dir: Path to the preprocessed data directory
        tokenizer_type: Optional hint for fallback size ('optimized', 'preserve', 'canonical')
    
    Returns:
        Vocabulary size
    """
    detected_tokenizer_type = tokenizer_type
    
    # Try config.json first
    config_path = os.path.join(data_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data_config = json.load(f)
        if 'vocab_size' in data_config:
            return data_config['vocab_size']
        # Extract tokenizer_type for fallback if not provided
        if detected_tokenizer_type is None:
            detected_tokenizer_type = data_config.get('tokenizer_type')
    
    # Try vocab.json
    vocab_path = os.path.join(data_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        return len(vocab)
    
    # Fallback based on tokenizer type
    fallback_sizes = {
        'optimized': 2500,
        'preserve': 10000,
        'canonical': 15000,
    }
    
    if detected_tokenizer_type in fallback_sizes:
        fallback = fallback_sizes[detected_tokenizer_type]
        print(f"Warning: Could not determine vocab_size from data. "
              f"Using fallback {fallback} for tokenizer_type='{detected_tokenizer_type}'")
        return fallback
    
    # Unknown tokenizer type - use conservative fallback with strong warning
    print("=" * 60)
    print("ERROR: Could not determine vocab_size and tokenizer_type is unknown!")
    print("This may cause embedding dimension mismatches.")
    print("Please ensure config.json or vocab.json exists in:", data_dir)
    print("Using fallback vocab_size=10000 (may be incorrect)")
    print("=" * 60)
    return 10000


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
    
    # Log embedding layer size for debugging dimension mismatches
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    emb_shape = actual_model.embedding.weight.shape
    print(f"Embedding layer: vocab_size={emb_shape[0]}, embed_dim={emb_shape[1]}")
    
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
    metric: str = 'f1',
    min_t: float = 0.2,
    max_t: float = 0.8,
    step: float = 0.01
) -> Tuple[float, float, List[Dict]]:
    """Find optimal threshold for classification.
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        metric: 'f1', 'precision', 'recall', 'balanced' (geometric mean of P and R),
                'mcc' (Matthews Correlation Coefficient), or 'youden' (TPR - FPR)
        min_t, max_t, step: Search range
    
    Returns:
        best_threshold, best_score, results_list
    """
    from sklearn.metrics import matthews_corrcoef
    
    thresholds = np.arange(min_t, max_t + step, step)
    results = []
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
            
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        if metric == 'f1':
            score = f1
        elif metric == 'precision':
            score = p
        elif metric == 'recall':
            score = r
        elif metric == 'balanced':
            score = np.sqrt(p * r) if p > 0 and r > 0 else 0.0
        elif metric == 'mcc':
            score = matthews_corrcoef(labels, preds)
        elif metric == 'youden':
            # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
            tn = np.sum((preds == 0) & (labels == 0))
            fp = np.sum((preds == 1) & (labels == 0))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = r + specificity - 1  # TPR + TNR - 1
        else:
            score = f1
            
        results.append({
            'threshold': float(thresh),
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'score': float(score)
        })
    
    if not results:
        return 0.5, 0.0, []
    
    best = max(results, key=lambda x: x['score'])
    return best['threshold'], best['score'], results


def get_metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute metrics at a specific threshold."""
    preds = (probs >= threshold).astype(int)
    return {
        'threshold': threshold,
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'accuracy': accuracy_score(labels, preds),
    }


def plot_threshold_analysis(results: List[Dict], optimal_threshold: float, save_path: str = None):
    """Plot precision, recall, F1 vs threshold."""
    if not results:
        return
    
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
    ax.plot(thresholds, f1s, 'r-', linewidth=2, label='F1 Score')
    
    ax.axvline(x=optimal_threshold, color='purple', linestyle='--', linewidth=2, 
               label=f'Optimal Threshold ({optimal_threshold:.2f})')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, 
               label='Default (0.5)')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision-Recall-F1 vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(thresholds), max(thresholds)])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved threshold analysis: {save_path}")
    
    # Only show in interactive environments (not headless servers)
    if plt.isinteractive() or 'inline' in plt.get_backend().lower():
        plt.show()
    plt.close()


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
        extended_token_type_ids = batch.get("extended_token_type_ids")
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device, non_blocking=True)
        if lengths is not None:
            lengths = lengths.to(device, non_blocking=True)
        if extended_token_type_ids is not None:
            extended_token_type_ids = extended_token_type_ids.to(device, non_blocking=True)
        
        b = input_ids.size(0)
        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum
        
        model(input_ids, attention_mask, vuln_features, lengths, extended_token_type_ids=extended_token_type_ids)
        n += b
    
    for module, mom in momenta.items():
        module.momentum = mom
    
    model.train(was_training)


# %% [markdown]
# ## 5. Training Loop

# %%
def _prepare_batch(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Prepare batch tensors for forward pass.
    
    Returns:
        Tuple of (input_ids, attention_mask, labels, vuln_features, lengths, extended_token_type_ids)
    """
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)
    vuln_features = batch.get('vuln_features')
    lengths = batch['lengths'].to(device, non_blocking=True)
    extended_token_type_ids = batch.get('extended_token_type_ids')
    
    if vuln_features is not None:
        vuln_features = vuln_features.to(device, non_blocking=True)
    
    if extended_token_type_ids is not None:
        extended_token_type_ids = extended_token_type_ids.to(device, non_blocking=True)
    
    return input_ids, attention_mask, labels, vuln_features, lengths, extended_token_type_ids


def _forward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vuln_features: Optional[torch.Tensor],
    lengths: torch.Tensor,
    config: BaselineConfig,
    extended_token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run forward pass with AMP."""
    with autocast(device_type='cuda', enabled=config.use_amp):
        outputs = model(
            input_ids, 
            attention_mask, 
            vuln_features, 
            lengths,
            extended_token_type_ids=extended_token_type_ids,
        )
    return outputs.logits


def _compute_step_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    vuln_features: Optional[torch.Tensor],
    lengths: torch.Tensor,
    extended_token_type_ids: Optional[torch.Tensor],
    criterion: nn.Module,
    apply_contrastive: bool,
    contrastive_criterion: Optional[nn.Module] = None,
    contrastive_weight: float = 0.5,
    use_supcon: bool = True,
) -> Dict[str, Any]:
    """Compute forward pass and losses for a single training step.
    
    Memory-efficient contrastive learning: calls forward() twice (original + augmented)
    instead of computing both views in a single pass. This releases the activation
    graph of the first view before computing the second, reducing peak VRAM usage.
    
    Args:
        model: The model to run forward pass on
        input_ids, attention_mask, labels, vuln_features, lengths, extended_token_type_ids: Batch tensors
        criterion: Main classification loss function
        apply_contrastive: Whether to apply contrastive learning this step
        contrastive_criterion: Contrastive loss function (SupConLoss or SimCLRLoss)
        contrastive_weight: Weight for contrastive loss term
        use_supcon: If True, use SupCon; otherwise use SimCLR
    
    Returns:
        Dict with 'loss', 'cls_loss', 'con_loss', and 'logits'
    """
    if apply_contrastive:
        # Forward pass 1: Original view
        outputs = model(
            input_ids, 
            attention_mask, 
            vuln_features, 
            lengths,
            extended_token_type_ids=extended_token_type_ids,
            return_embeddings=True,
            return_projections=True,
        )
        logits = outputs.logits
        projections = outputs.projections
        
        # Classification loss (only on original view)
        cls_loss = criterion(logits, labels)
        
        # Forward pass 2: Augmented view (separate call = separate activation graph)
        # Handle DataParallel wrapper
        base_model = model.module if hasattr(model, 'module') else model
        aug_input_ids = base_model.get_augmented_input(input_ids, attention_mask)
        aug_outputs = model(
            aug_input_ids, 
            attention_mask, 
            vuln_features, 
            lengths,
            extended_token_type_ids=extended_token_type_ids,
            return_embeddings=True,
            return_projections=True,
        )
        aug_projections = aug_outputs.projections
        
        # Contrastive loss
        if use_supcon:
            # SupCon with 2 views: stack as [B, 2, D] for proper multi-view handling
            # This tells SupConLoss that sample i has 2 views that should be treated as positives
            stacked_projections = torch.stack([projections, aug_projections], dim=1)  # [B, 2, D]
            con_loss = contrastive_criterion(stacked_projections, labels)
        else:
            # SimCLR: compare original vs augmented views
            con_loss = contrastive_criterion(projections, aug_projections)
        
        total_loss = cls_loss + contrastive_weight * con_loss
        return {
            'loss': total_loss,
            'cls_loss': cls_loss.item(),
            'con_loss': con_loss.item(),
            'logits': logits,
        }
    else:
        outputs = model(
            input_ids, 
            attention_mask, 
            vuln_features, 
            lengths,
            extended_token_type_ids=extended_token_type_ids,
        )
        logits = outputs.logits
        loss = criterion(logits, labels)
        return {
            'loss': loss,
            'cls_loss': loss.item(),
            'con_loss': 0.0,
            'logits': logits,
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    config: BaselineConfig,
    device: torch.device,
    compute_accurate_metrics: bool = True,
    epoch: int = 0,
    contrastive_criterion: nn.Module = None,
) -> Dict[str, float]:
    """Single training epoch with optional contrastive learning.
    
    Args:
        compute_accurate_metrics: If True, compute metrics in eval mode at end of epoch.
            This adds ~10-15% time but gives accurate train F1/AUC comparable to val metrics.
            If False, uses approximate metrics collected during training (with dropout ON).
        epoch: Current epoch number (for contrastive warmup)
        contrastive_criterion: Contrastive loss function (SupConLoss or SimCLRLoss)
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_con_loss = 0.0
    # Always collect approximate metrics during training (for fallback)
    approx_probs, approx_labels = [], []
    
    # Contrastive learning settings
    use_contrastive = getattr(config, 'use_contrastive', False)
    contrastive_warmup = getattr(config, 'contrastive_warmup_epochs', 2)
    contrastive_weight = getattr(config, 'contrastive_weight', 0.5)
    use_supcon = getattr(config, 'use_supcon', True)
    
    # Only use contrastive after warmup
    apply_contrastive = use_contrastive and epoch >= contrastive_warmup and contrastive_criterion is not None
    
    if apply_contrastive and epoch == contrastive_warmup:
        print(f"  [Contrastive] Starting contrastive learning (weight={contrastive_weight})")
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_ids, attention_mask, labels, vuln_features, lengths, extended_token_type_ids = _prepare_batch(batch, device)
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            step_result = _compute_step_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                vuln_features=vuln_features,
                lengths=lengths,
                extended_token_type_ids=extended_token_type_ids,
                criterion=criterion,
                apply_contrastive=apply_contrastive,
                contrastive_criterion=contrastive_criterion,
                contrastive_weight=contrastive_weight,
                use_supcon=use_supcon,
            )
            loss = step_result['loss']
            logits = step_result['logits']
            total_cls_loss += step_result['cls_loss']
            total_con_loss += step_result['con_loss']
            
            loss = loss / config.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.accumulation_steps
        
        # Collect approximate metrics (with dropout ON)
        probs = get_probs_from_logits(logits.detach()).cpu().numpy()
        approx_probs.extend(probs)
        approx_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    # Compute accurate metrics in eval mode (no dropout)
    if compute_accurate_metrics:
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, labels_batch, vuln_features, lengths, extended_token_type_ids = _prepare_batch(batch, device)
                logits = _forward_pass(model, input_ids, attention_mask, vuln_features, lengths, config, extended_token_type_ids)
                
                probs = get_probs_from_logits(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels_batch.cpu().numpy())
        
        model.train()  # Restore train mode
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
    else:
        # Use approximate metrics collected during training (with dropout ON)
        # These will be slightly lower than eval metrics but still valid for tracking
        all_probs = np.array(approx_probs)
        all_labels = np.array(approx_labels)
    
    all_preds = (all_probs >= 0.5).astype(int)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    result = {
        'loss': avg_loss,
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': auc,
    }
    
    # Add contrastive loss info if applicable
    if use_contrastive:
        n_batches = len(loader) if len(loader) > 0 else 1
        result['cls_loss'] = total_cls_loss / n_batches
        result['con_loss'] = total_con_loss / n_batches if apply_contrastive else 0.0
    
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: BaselineConfig,
    device: torch.device,
    return_threshold_analysis: bool = False,
    use_fixed_threshold: bool = False,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate model with optimal threshold search.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation
        criterion: Loss function
        config: Configuration with threshold settings
        device: Device to use
        return_threshold_analysis: If True, include threshold analysis results
        use_fixed_threshold: If True, use fixed 0.5 threshold (Oracle: reduces noise during training)
        temperature: Temperature for probability calibration (default 1.0 = no scaling)
    
    Returns:
        Dict with metrics, optionally including threshold analysis
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        vuln_features = batch.get('vuln_features')
        lengths = batch['lengths'].to(device, non_blocking=True)
        extended_token_type_ids = batch.get('extended_token_type_ids')
        
        if vuln_features is not None:
            vuln_features = vuln_features.to(device, non_blocking=True)
        
        if extended_token_type_ids is not None:
            extended_token_type_ids = extended_token_type_ids.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            outputs = model(
                input_ids, 
                attention_mask, 
                vuln_features, 
                lengths,
                extended_token_type_ids=extended_token_type_ids,
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = get_probs_from_logits(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Apply temperature scaling if provided
    if temperature != 1.0:
        # Convert to logits, scale, convert back to probs
        # probs = sigmoid(logits) -> logits = logit(probs)
        eps = 1e-7
        logits = np.log(all_probs + eps) - np.log(1 - all_probs + eps)
        scaled_logits = logits / temperature
        all_probs = 1 / (1 + np.exp(-scaled_logits))
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Happens when only one class is present in batch
        auc = 0.5
    
    # Oracle: Use fixed threshold during training to avoid noise
    if use_fixed_threshold:
        best_t = 0.5
        best_score = None
        threshold_results = None
    else:
        optimization_metric = getattr(config, 'threshold_optimization_metric', 'f1')
        
        best_t, best_score, threshold_results = find_optimal_threshold(
            all_labels, all_probs,
            metric=optimization_metric,
            min_t=config.threshold_min,
            max_t=config.threshold_max,
            step=config.threshold_step
        )
    
    preds_optimal = (all_probs >= best_t).astype(int)
    
    default_metrics = get_metrics_at_threshold(all_labels, all_probs, 0.5)
    
    result = {
        'loss': avg_loss,
        'f1': f1_score(all_labels, preds_optimal, zero_division=0),
        'precision': precision_score(all_labels, preds_optimal, zero_division=0),
        'recall': recall_score(all_labels, preds_optimal, zero_division=0),
        'auc': auc,
        'threshold': best_t,
        'probs': all_probs,
        'labels': all_labels,
        'default_f1': default_metrics['f1'],
        'default_precision': default_metrics['precision'],
        'default_recall': default_metrics['recall'],
        'f1_improvement': f1_score(all_labels, preds_optimal, zero_division=0) - default_metrics['f1'],
    }
    
    if return_threshold_analysis:
        result['threshold_analysis'] = threshold_results
    
    return result


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% [markdown]
# ## 6. Visualization Functions

# %%
def plot_training_history(history: Dict[str, List[float]], seed: int, save_path: str = None):
    """
    Plot training history: loss, F1, and AUC over epochs.
    
    Args:
        history: Dict with keys 'train_loss', 'train_f1', 'val_f1', 'val_auc'
        seed: Seed number for title
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot Loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot F1
    ax2 = axes[1]
    ax2.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax2.plot(epochs, history['val_f1'], 'r--', label='Val F1', linewidth=2)
    best_epoch = np.argmax(history['val_f1']) + 1
    best_f1 = max(history['val_f1'])
    ax2.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label=f'Best (ep={best_epoch})')
    ax2.scatter([best_epoch], [best_f1], color='g', s=100, zorder=5, marker='*')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('Training & Validation F1', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Plot AUC
    ax3 = axes[2]
    if 'train_auc' in history:
        ax3.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax3.plot(epochs, history['val_auc'], 'r--', label='Val AUC', linewidth=2)
    best_auc_epoch = np.argmax(history['val_auc']) + 1
    best_auc = max(history['val_auc'])
    ax3.scatter([best_auc_epoch], [best_auc], color='g', s=100, zorder=5, marker='*')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('AUC', fontsize=11)
    ax3.set_title('Validation AUC', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History (Seed {seed})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved training history plot: {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix(
    labels: np.ndarray, 
    preds: np.ndarray, 
    threshold: float,
    seed: int = None,
    title_suffix: str = "",
    save_path: str = None
):
    """
    Plot confusion matrix with detailed annotations.
    
    Args:
        labels: Ground truth labels
        preds: Predicted labels (binary)
        threshold: Threshold used for prediction
        seed: Optional seed number for title
        title_suffix: Additional text for title (e.g., "Test Set")
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(labels, preds)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=False,  # We'll add custom annotations
        fmt='d', 
        cmap='Blues',
        xticklabels=['Non-Vulnerable (0)', 'Vulnerable (1)'],
        yticklabels=['Non-Vulnerable (0)', 'Vulnerable (1)'],
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    # Add custom annotations with count and percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f'{count}\n({percent:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text, 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Build title
    title = "Confusion Matrix"
    if title_suffix:
        title += f" - {title_suffix}"
    if seed is not None:
        title += f" (Seed {seed})"
    title += f"\nThreshold: {threshold:.2f}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add metrics as text - use sklearn for consistency with EvalMetrics
    metrics_text = (
        f'Accuracy: {accuracy_score(labels, preds):.4f}\n'
        f'Precision: {precision_score(labels, preds, zero_division=0):.4f}\n'
        f'Recall: {recall_score(labels, preds, zero_division=0):.4f}\n'
        f'F1 Score: {f1_score(labels, preds, zero_division=0):.4f}'
    )
    ax.text(2.5, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved confusion matrix: {save_path}")
    
    plt.show()
    plt.close()


def plot_multi_seed_comparison(all_histories: List[Dict], seeds: List[int], save_path: str = None):
    """
    Plot comparison of training curves across multiple seeds.
    
    Args:
        all_histories: List of history dicts from each seed
        seeds: List of seed values
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # Plot Val F1 across seeds
    ax1 = axes[0]
    for i, (history, seed) in enumerate(zip(all_histories, seeds)):
        epochs = range(1, len(history['val_f1']) + 1)
        ax1.plot(epochs, history['val_f1'], color=colors[i], linewidth=2, 
                label=f'Seed {seed} (best={max(history["val_f1"]):.4f})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Validation F1', fontsize=11)
    ax1.set_title('Validation F1 Across Seeds', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot Val AUC across seeds
    ax2 = axes[1]
    for i, (history, seed) in enumerate(zip(all_histories, seeds)):
        epochs = range(1, len(history['val_auc']) + 1)
        ax2.plot(epochs, history['val_auc'], color=colors[i], linewidth=2,
                label=f'Seed {seed} (best={max(history["val_auc"]):.4f})')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation AUC', fontsize=11)
    ax2.set_title('Validation AUC Across Seeds', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Seed Training Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved multi-seed comparison: {save_path}")
    
    plt.show()
    plt.close()


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
    plot_history: bool = True,
) -> Tuple[EvalMetrics, nn.Module, Dict[str, List[float]]]:
    """
    Train model with a single seed.
    
    Returns:
        metrics: EvalMetrics from best checkpoint
        model: Trained model (SWA if enabled, else best checkpoint)
        history: Training history dict with train_loss, train_f1, val_f1, val_auc, val_loss
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
        focal_gamma=getattr(config, 'focal_gamma', 2.0),
        focal_alpha=getattr(config, 'focal_alpha', None),
    ).to(DEVICE)
    
    # Contrastive loss function (if enabled)
    contrastive_criterion = None
    use_contrastive = getattr(config, 'use_contrastive', False)
    if use_contrastive:
        use_supcon = getattr(config, 'use_supcon', True)
        contrastive_temp = getattr(config, 'contrastive_temperature', 0.07)
        if use_supcon:
            contrastive_criterion = SupConLoss(temperature=contrastive_temp)
            if verbose:
                print(f"  [Contrastive] SupConLoss enabled (temp={contrastive_temp})")
        else:
            contrastive_criterion = SimCLRLoss(temperature=contrastive_temp)
            if verbose:
                print(f"  [Contrastive] SimCLRLoss enabled (temp={contrastive_temp})")
    
    # Optimizer (Word2Vec removed - no differential LR needed)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Scheduler
    use_warmup = getattr(config, 'use_warmup', False)
    warmup_epochs = getattr(config, 'warmup_epochs', 2)
    
    if config.scheduler_type == 'cosine':
        if use_warmup:
            # Cosine with warmup
            from torch.optim.lr_scheduler import LambdaLR
            
            def warmup_cosine_schedule(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (config.max_epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
        else:
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
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr, anneal_epochs=2)
    
    # Training loop with history tracking
    best_score, best_epoch = 0.0, 0
    best_state = None
    best_metrics = None
    
    history = {
        'train_loss': [],
        'train_f1': [],
        'train_auc': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': [],
    }
    
    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        
        # Train - only compute accurate metrics every 5 epochs to save time
        compute_accurate = (epoch % 5 == 0) or (epoch == config.max_epochs)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, config, DEVICE, 
                                    compute_accurate_metrics=compute_accurate,
                                    epoch=epoch,
                                    contrastive_criterion=contrastive_criterion)
        
        # Validate - Oracle: Use fixed threshold during training to reduce noise
        use_fixed = getattr(config, 'report_fixed_threshold_during_training', True)
        val_metrics = evaluate(model, val_loader, criterion, config, DEVICE, 
                              use_fixed_threshold=use_fixed)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Record contrastive loss if applicable
        if use_contrastive and 'con_loss' in train_metrics:
            if 'train_con_loss' not in history:
                history['train_con_loss'] = []
            history['train_con_loss'].append(train_metrics['con_loss'])
        
        # Get current LR for logging
        current_lr = optimizer.param_groups[0]['lr']
        
        if verbose:
            con_info = f" ConL={train_metrics.get('con_loss', 0):.3f}" if use_contrastive and train_metrics.get('con_loss', 0) > 0 else ""
            print(
                f"  Ep {epoch:2d}/{config.max_epochs} ({epoch_time:.0f}s) LR={current_lr:.2e} | "
                f"Train: L={train_metrics['loss']:.3f} F1={train_metrics['f1']:.3f}{con_info} | "
                f"Val: F1={val_metrics['f1']:.3f} AUC={val_metrics['auc']:.3f} T={val_metrics['threshold']:.2f}"
            )
        
        # Scheduler step
        if config.use_swa and epoch >= config.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif config.scheduler_type == 'plateau':
            scheduler.step(val_metrics['auc'])
        else:
            scheduler.step()
        
        # Early stopping metric selection - Oracle: AUC is more stable than F1
        es_metric = getattr(config, 'early_stopping_metric', 'auc')
        if es_metric == 'auc':
            current_metric = val_metrics['auc']
        else:
            current_metric = val_metrics['f1']
        
        # Save best (always track best by the early stopping metric)
        if current_metric > best_score:
            best_score = current_metric
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
        
        if early_stopping(current_metric):
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best {es_metric}={best_score:.4f})")
            break
    
    # SWA finalization
    if config.use_swa and swa_model is not None:
        update_bn_for_swa(train_loader, swa_model, DEVICE)
        swa_metrics = evaluate(swa_model, val_loader, criterion, config, DEVICE)
        
        if verbose:
            print(f"  [SWA] Val: F1={swa_metrics['f1']:.4f} AUC={swa_metrics['auc']:.4f}")
        
        if swa_metrics['f1'] > best_score:
            best_metrics = swa_metrics
            final_model = swa_model
            if verbose:
                print(f"  [SWA] Using SWA model (F1={swa_metrics['f1']:.4f} > {best_score:.4f})")
        else:
            model.load_state_dict(best_state)
            final_model = model
    else:
        model.load_state_dict(best_state)
        final_model = model
    
    # Temperature scaling - Oracle: Post-hoc calibration helps precision
    temperature = 1.0
    if getattr(config, 'use_temperature_scaling', False):
        if verbose:
            print("  Fitting temperature scaling...")
        temp_scaler = TemperatureScaling().to(DEVICE)
        final_base = final_model.module if hasattr(final_model, 'module') else final_model
        temperature = temp_scaler.fit(final_base, val_loader, config, DEVICE)
    
    # Re-evaluate with optimal threshold (not fixed) after training
    final_metrics = evaluate(final_model, val_loader, criterion, config, DEVICE, 
                            use_fixed_threshold=False, temperature=temperature)
    
    eval_metrics = EvalMetrics(
        f1=final_metrics['f1'],
        precision=final_metrics['precision'],
        recall=final_metrics['recall'],
        auc=final_metrics['auc'],
        accuracy=accuracy_score(final_metrics['labels'], (final_metrics['probs'] >= final_metrics['threshold']).astype(int)),
        threshold=final_metrics['threshold'],
    )
    
    # Plot training history
    if plot_history:
        history_plot_path = os.path.join(PLOT_DIR, f'training_history_seed{seed}.png')
        plot_training_history(history, seed, save_path=history_plot_path)
    
    return eval_metrics, final_model, history


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
    all_histories = []
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        metrics, model, history = train_single_seed(
            config, seed, train_loader, val_loader, pretrained_embedding,
            plot_history=True
        )
        all_results.append(metrics)
        all_models.append(model)
        all_histories.append(history)
        
        print(f"  → F1={metrics.f1:.4f}, AUC={metrics.auc:.4f}, T={metrics.threshold:.2f}")
    
    # Plot multi-seed comparison
    comparison_path = os.path.join(PLOT_DIR, 'multi_seed_comparison.png')
    plot_multi_seed_comparison(all_histories, seeds, save_path=comparison_path)
    
    # Aggregate results
    aggregated = aggregate_results(all_results, config_name="multi_seed")
    
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(aggregated)
    
    # Test set evaluation with best model (highest val F1)
    best_idx = np.argmax([r.f1 for r in all_results])
    best_model = all_models[best_idx]
    best_seed = seeds[best_idx]
    
    print(f"\n--- Test Set Evaluation (using seed {best_seed}) ---")
    
    # Create criterion for test eval
    test_labels = test_loader.dataset.labels
    pos_weight = get_pos_weight_for_config(config, test_labels)
    criterion = SimplifiedLoss(
        loss_type=config.loss_type, 
        pos_weight=pos_weight,
        focal_gamma=getattr(config, 'focal_gamma', 2.0),
        focal_alpha=getattr(config, 'focal_alpha', None),
    ).to(DEVICE)
    
    test_metrics = evaluate(best_model, test_loader, criterion, config, DEVICE)
    print(f"Test: F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} "
          f"R={test_metrics['recall']:.4f} AUC={test_metrics['auc']:.4f}")
    
    # Plot confusion matrix for test set
    test_preds = (test_metrics['probs'] >= test_metrics['threshold']).astype(int)
    cm_path = os.path.join(PLOT_DIR, f'confusion_matrix_test_seed{best_seed}.png')
    plot_confusion_matrix(
        test_metrics['labels'], 
        test_preds, 
        test_metrics['threshold'],
        seed=best_seed,
        title_suffix="Test Set",
        save_path=cm_path
    )
    
    # Save best model
    save_path = os.path.join(MODEL_DIR, f'best_model_seed{best_seed}.pt')
    base_model = best_model.module if isinstance(best_model, nn.DataParallel) else best_model
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'config': config.to_dict(),
        'seed': best_seed,
        'val_f1': all_results[best_idx].f1,
        'val_auc': all_results[best_idx].auc,
        'threshold': all_results[best_idx].threshold,
        'test_metrics': {
            'f1': test_metrics['f1'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'auc': test_metrics['auc'],
        }
    }, save_path)
    print(f"\nSaved best model to {save_path}")
    
    # Also save all seed models for ensemble
    for i, (model, seed) in enumerate(zip(all_models, seeds)):
        model_path = os.path.join(MODEL_DIR, f'model_seed{seed}.pt')
        base_m = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'model_state_dict': base_m.state_dict(),
            'config': config.to_dict(),
            'seed': seed,
            'val_f1': all_results[i].f1,
            'val_auc': all_results[i].auc,
        }, model_path)
    print(f"Saved all {n_seeds} models to {MODEL_DIR}")
    
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
        config_name: One of "baseline", "improved", "conservative", "recall_focused", 
                     "precision_focused", "large", "focal", "contrastive", "simclr"
        n_seeds: Number of seeds for multi-seed evaluation
        data_dir: Path to preprocessed data
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Get config
    config_getters = {
        "baseline": get_baseline_config,
        "improved": get_improved_config,
        "conservative": get_conservative_config,
        "recall_focused": get_recall_focused_config,
        "precision_focused": get_precision_focused_config,
        "balanced_pr": get_balanced_pr_config,
        "large": get_large_config,
        "focal": get_focal_config,
        "no_pretrained": get_no_pretrained_config,
        "contrastive": get_contrastive_config,
        "simclr": get_simclr_config,
    }
    
    if config_name not in config_getters:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(config_getters.keys())}")
    
    config = config_getters[config_name]()
    
    # Load vocab_size from data and update config
    actual_vocab_size = load_vocab_size_from_data(data_dir)
    if actual_vocab_size != config.vocab_size:
        print(f"Updating vocab_size: {config.vocab_size} -> {actual_vocab_size}")
        config = config.override(vocab_size=actual_vocab_size)
    
    print(f"{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*60}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  loss_type: {config.loss_type}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  scheduler: {config.scheduler_type} (warmup={getattr(config, 'use_warmup', False)})")
    print(f"  use_swa: {config.use_swa} (start={config.swa_start_epoch})")
    print(f"  use_token_augmentation: {config.use_token_augmentation}")
    print(f"  use_temperature_scaling: {getattr(config, 'use_temperature_scaling', False)}")
    print(f"  vuln_feature_dropout: {config.vuln_feature_dropout}")
    # Contrastive learning info
    use_contrastive = getattr(config, 'use_contrastive', False)
    if use_contrastive:
        print(f"  use_contrastive: {use_contrastive}")
        print(f"    contrastive_weight: {config.contrastive_weight}")
        print(f"    contrastive_temperature: {config.contrastive_temperature}")
        print(f"    use_supcon: {config.use_supcon}")
        print(f"    warmup_epochs: {config.contrastive_warmup_epochs}")
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
        seed = config.ensemble_base_seed
        metrics, model, history = train_single_seed(
            config, 
            seed, 
            train_loader, 
            val_loader, 
            pretrained_embedding,
            plot_history=True
        )
        print(f"\nFinal: F1={metrics.f1:.4f}, AUC={metrics.auc:.4f}")
        
        # Evaluate on test set and plot confusion matrix
        test_labels = test_loader.dataset.labels
        pos_weight = get_pos_weight_for_config(config, test_labels)
        criterion = SimplifiedLoss(
            loss_type=config.loss_type, 
            pos_weight=pos_weight,
            focal_gamma=getattr(config, 'focal_gamma', 2.0),
            focal_alpha=getattr(config, 'focal_alpha', None),
        ).to(DEVICE)
        test_metrics = evaluate(model, test_loader, criterion, config, DEVICE, return_threshold_analysis=True)
        
        print(f"\n{'='*60}")
        print("TEST SET EVALUATION - Threshold Analysis")
        print(f"{'='*60}")
        print(f"  Optimal threshold: {test_metrics['threshold']:.3f}")
        print(f"  Optimized:  F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}")
        print(f"  Default@0.5: F1={test_metrics['default_f1']:.4f} P={test_metrics['default_precision']:.4f} R={test_metrics['default_recall']:.4f}")
        print(f"  F1 improvement: {test_metrics['f1_improvement']:+.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"{'='*60}")
        
        if 'threshold_analysis' in test_metrics:
            threshold_plot_path = os.path.join(PLOT_DIR, f'threshold_analysis_seed{seed}.png')
            plot_threshold_analysis(
                test_metrics['threshold_analysis'],
                test_metrics['threshold'],
                save_path=threshold_plot_path
            )
        
        # Plot confusion matrix for test set
        test_preds = (test_metrics['probs'] >= test_metrics['threshold']).astype(int)
        cm_path = os.path.join(PLOT_DIR, f'confusion_matrix_test_seed{seed}.png')
        plot_confusion_matrix(
            test_metrics['labels'], 
            test_preds, 
            test_metrics['threshold'],
            seed=seed,
            title_suffix="Test Set",
            save_path=cm_path
        )
        
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
    
    parser = argparse.ArgumentParser(description="Devign Training V2 - With Oracle Improvements + Contrastive Learning")
    parser.add_argument("--config", type=str, default="baseline",
                        choices=["baseline", "improved", "conservative", "recall_focused", 
                                "precision_focused", "balanced_pr", "large", "focal", "no_pretrained",
                                "contrastive", "simclr"],
                        help="Config preset to use (default: baseline with all Oracle fixes)")
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
# Uncomment the line below to run training with Oracle-recommended improvements
# main(config_name="baseline", n_seeds=3)

# %%
# === CONTRASTIVE LEARNING EXAMPLES ===
# 
# Option 1: Use pre-defined contrastive config (SupCon)
# main(config_name="contrastive", n_seeds=3)
#
# Option 2: Use SimCLR config (self-supervised style)
# main(config_name="simclr", n_seeds=3)
#
# Option 3: Manually enable contrastive on any config:
# config = get_baseline_config()
# config = config.override(
#     use_contrastive=True,
#     contrastive_weight=0.5,
#     contrastive_temperature=0.07,
#     use_supcon=True,
#     contrastive_warmup_epochs=2,
# )

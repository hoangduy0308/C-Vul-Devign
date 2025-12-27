"""
Simplified Training Configuration for Devign Vulnerability Detection.

Based on Oracle recommendations:
- 1 baseline + 2-3 variants (instead of 20+ classes)
- Single loss strategy: BCEWithLogitsLoss(pos_weight)
- Clean ablation-ready design

Author: Refactored from 02_training.py configs
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import json


@dataclass
class BaselineConfig:
    """
    Baseline configuration - the single source of truth.
    
    This is the ONLY config you need for most experiments.
    Change individual parameters via overrides, not inheritance.
    
    Architecture: Embedding → BiGRU (2 layers) → Attention → Classifier
    Loss: BCEWithLogitsLoss with pos_weight (single strategy, no stacking)
    """
    
    # ============= MODEL ARCHITECTURE =============
    vocab_size: int = 30000
    embed_dim: int = 128
    hidden_dim: int = 160          # BiGRU output = 320 (bidirectional)
    num_layers: int = 2
    bidirectional: bool = True
    
    # Attention
    use_multihead_attention: bool = True
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # LayerNorm
    use_layer_norm: bool = True
    
    # ============= REGULARIZATION =============
    # Oracle: Pick ONE dropout level, don't stack multiple
    embedding_dropout: float = 0.15
    rnn_dropout: float = 0.25
    classifier_dropout: float = 0.25
    
    # Weight decay
    weight_decay: float = 1e-4
    
    # ============= LOSS FUNCTION =============
    # Oracle: Use SINGLE loss strategy - don't stack multiple
    # Options: "bce", "bce_weighted", "focal", "focal_alpha"
    loss_type: str = "bce_weighted"
    
    # For bce_weighted: pos_weight will be computed from data
    # pos_weight = n_negative / n_positive (auto-computed in trainer)
    
    # Label smoothing: mild or none (Oracle recommends 0.0-0.03)
    label_smoothing: float = 0.0  # OFF by default for clean baseline
    
    # pos_weight override: Set to 1.0 for balanced data (ratio ~1.18:1)
    # Weighting hurts precision with nearly balanced classes
    pos_weight_override: Optional[float] = 1.0
    
    # Focal Loss parameters (used when loss_type="focal" or "focal_alpha")
    # gamma: Focusing parameter. 0=BCE, 2=typical, 5=aggressive focus on hard examples
    focal_gamma: float = 2.0
    # alpha: Class weight for positives (0-1). 
    #   - alpha < 0.5: down-weight positives (use when model over-predicts positive)
    #   - alpha > 0.5: up-weight positives (use when model under-predicts positive)
    #   - None: no class weighting (pure focal loss)
    focal_alpha: Optional[float] = None
    
    # ============= TRAINING =============
    batch_size: int = 128
    accumulation_steps: int = 1
    learning_rate: float = 3e-4
    max_epochs: int = 25
    grad_clip: float = 1.0
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    
    # Scheduler
    scheduler_type: str = "plateau"  # Options: "plateau", "cosine"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # ============= SWA =============
    # Oracle: SWA alone can replace part of ensemble
    # Disabled: Early stopping triggers at epoch 9-12, but SWA starts at epoch 15
    use_swa: bool = False
    swa_start_epoch: int = 15
    swa_lr: float = 5e-5
    
    # ============= ENSEMBLE =============
    # Oracle: 2-3 seeds is enough, not 5-7
    ensemble_size: int = 3
    ensemble_base_seed: int = 42
    
    # ============= DATA =============
    max_seq_length: int = 512
    num_workers: int = 2
    use_packed_sequences: bool = True
    
    # ============= PRETRAINED EMBEDDING =============
    use_pretrained_embedding: bool = True
    embedding_path: str = ""
    freeze_embedding: bool = False
    embedding_lr_scale: float = 0.1
    
    # ============= THRESHOLD =============
    use_optimal_threshold: bool = True
    threshold_min: float = 0.2
    threshold_max: float = 0.7
    threshold_step: float = 0.01
    threshold_optimization_metric: str = 'mcc'  # 'f1', 'precision', 'recall', 'balanced', 'mcc', 'youden'
    
    # ============= VULN FEATURES (optional) =============
    use_vuln_features: bool = True  # Enabled: handcrafted features help when tokens lack discriminative power
    vuln_feature_dim: int = 25
    vuln_feature_hidden_dim: int = 64
    vuln_feature_dropout: float = 0.2
    
    # ============= TOKEN AUGMENTATION =============
    use_token_augmentation: bool = False  # OFF by default
    token_dropout_prob: float = 0.05
    token_mask_prob: float = 0.03
    mask_token_id: int = 1
    
    # ============= TOKEN TYPE EMBEDDING =============
    use_token_type_embedding: bool = True  # Enable vulnerability-relevant token type embedding
    num_token_types: int = 16              # Number of extended token types (from token_types.py)
    token_type_embed_dim: int = 32         # Smaller than main embed_dim for efficiency
    
    # ============= CHECKPOINTING =============
    save_every: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to plain dict."""
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BaselineConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def override(self, **kwargs) -> "BaselineConfig":
        """Create a new config with specific overrides."""
        data = self.to_dict()
        data.update(kwargs)
        return BaselineConfig(**data)


# ============= PRESET VARIANTS =============
# These are the ONLY 3 variants you need

def get_baseline_config() -> BaselineConfig:
    """
    Baseline config - use this for most experiments.
    
    Loss: BCEWithLogitsLoss(pos_weight)
    No label smoothing, no focal loss, no class weight stacking.
    """
    return BaselineConfig()


def get_recall_focused_config() -> BaselineConfig:
    """
    Recall-focused variant.
    
    Use when you need higher recall (catch more vulnerabilities).
    Trade-off: Lower precision (more false positives).
    """
    return BaselineConfig().override(
        # Lower threshold range to catch more positives
        threshold_min=0.25,
        threshold_max=0.55,
        
        # Slightly less regularization
        classifier_dropout=0.30,
        rnn_dropout=0.30,
    )


def get_precision_focused_config() -> BaselineConfig:
    """
    Precision-focused variant.
    
    Use when you need higher precision (fewer false alarms).
    Trade-off: Lower recall (miss some vulnerabilities).
    """
    return BaselineConfig().override(
        # Higher threshold range to be more selective
        threshold_min=0.45,
        threshold_max=0.75,
        
        # More regularization for conservative predictions
        classifier_dropout=0.40,
        rnn_dropout=0.40,
        
        # Mild label smoothing for calibration
        label_smoothing=0.03,
    )


def get_large_config() -> BaselineConfig:
    """
    Larger model variant.
    
    Use only if baseline underfits (train metrics >> val metrics).
    """
    return BaselineConfig().override(
        hidden_dim=192,
        num_attention_heads=6,
        batch_size=96,  # Reduce for memory
        
        # Stronger regularization for larger model
        classifier_dropout=0.40,
        rnn_dropout=0.40,
    )


def get_focal_config() -> BaselineConfig:
    """
    Focal Loss focused config.
    
    DEPRECATED: Use BCE with class weights instead for nearly-balanced data.
    Focal loss with wrong alpha can cause model to collapse to constant predictions.
    
    Only use when:
    - Dataset is severely imbalanced (>5:1 ratio)
    - Model has confirmed high recall but very low precision
    """
    return BaselineConfig().override(
        # Use BCE for balanced data first, focal only for severe imbalance
        loss_type="bce_weighted",
        pos_weight_override=1.0,  # Data is balanced (45.8% / 54.2%)
        
        # Threshold optimization with MCC to prevent all-positive collapse
        threshold_optimization_metric='mcc',
        threshold_min=0.30,
        threshold_max=0.70,
    )


# ============= ABLATION HELPERS =============

def create_ablation_configs(base: BaselineConfig = None) -> Dict[str, BaselineConfig]:
    """
    Create configs for systematic ablation study.
    
    Oracle recommends: Change ONE thing at a time.
    
    Returns dict of {experiment_name: config}
    """
    if base is None:
        base = BaselineConfig()
    
    return {
        # Baseline
        "baseline": base,
        
        # Loss ablations
        "focal_only": base.override(loss_type="focal"),
        "focal_alpha_025": base.override(loss_type="focal_alpha", focal_alpha=0.25),
        "focal_alpha_035": base.override(loss_type="focal_alpha", focal_alpha=0.35),
        "focal_gamma_1": base.override(loss_type="focal", focal_gamma=1.0),
        "focal_gamma_3": base.override(loss_type="focal", focal_gamma=3.0),
        "label_smoothing_03": base.override(label_smoothing=0.03),
        
        # Dropout ablations
        "dropout_low": base.override(classifier_dropout=0.25, rnn_dropout=0.25),
        "dropout_high": base.override(classifier_dropout=0.45, rnn_dropout=0.45),
        
        # SWA ablations
        "no_swa": base.override(use_swa=False),
        "swa_early": base.override(swa_start_epoch=10),
        
        # Model size ablations
        "small_model": base.override(hidden_dim=128, num_attention_heads=2),
        "large_model": base.override(hidden_dim=192, num_attention_heads=6),
        
        # Token augmentation
        "with_augmentation": base.override(
            use_token_augmentation=True,
            token_dropout_prob=0.05,
            token_mask_prob=0.03
        ),
    }


# ============= MULTI-SEED RUNNER =============

@dataclass
class AblationResult:
    """Result from a single ablation run."""
    config_name: str
    seed: int
    f1: float
    precision: float
    recall: float
    auc: float
    threshold: float


def get_seeds_for_evaluation(n_seeds: int = 3, base_seed: int = 42) -> List[int]:
    """
    Get seed list for multi-seed evaluation.
    
    Oracle recommends: Report mean ± std across 3+ seeds.
    """
    return [base_seed + i * 1000 for i in range(n_seeds)]


# ============= CONFIG COMPARISON =============

def compare_configs(config1: BaselineConfig, config2: BaselineConfig) -> Dict[str, tuple]:
    """
    Compare two configs and return differences.
    
    Useful for documenting ablation changes.
    """
    d1 = config1.to_dict()
    d2 = config2.to_dict()
    
    diffs = {}
    for key in d1:
        if d1[key] != d2[key]:
            diffs[key] = (d1[key], d2[key])
    
    return diffs


if __name__ == "__main__":
    # Demo: Print baseline config
    config = get_baseline_config()
    print("=== Baseline Config ===")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\n=== Ablation Configs ===")
    ablations = create_ablation_configs()
    for name, cfg in ablations.items():
        diffs = compare_configs(config, cfg)
        if diffs:
            print(f"\n{name}:")
            for k, (v1, v2) in diffs.items():
                print(f"  {k}: {v1} → {v2}")
        else:
            print(f"\n{name}: (same as baseline)")

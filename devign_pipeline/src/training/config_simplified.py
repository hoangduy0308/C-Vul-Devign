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
    
    # Weight decay - Oracle recommends 1e-3 to 1e-2 for better generalization
    # NOTE: 1e-3 is higher than typical 1e-4, but Devign dataset is nearly balanced (45.8%/54.2%)
    # and we use bce_weighted loss with pos_weight to handle any class imbalance.
    # If underfitting occurs on minority class, reduce to 5e-4.
    weight_decay: float = 1e-3
    
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
    early_stopping_metric: str = "auc"  # "auc" or "f1" - Oracle: AUC is more stable
    
    # Scheduler - Oracle: cosine decay is less reactive/noisy on small val sets
    scheduler_type: str = "cosine"  # Options: "plateau", "cosine"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Learning rate warmup - Oracle: helps training stability
    use_warmup: bool = True
    warmup_epochs: int = 2  # ~5-10% of training
    
    # Mixed precision
    use_amp: bool = True
    
    # ============= SWA =============
    # Oracle: SWA alone can replace part of ensemble
    # Start SWA at epoch 8-10 where best performance typically occurs
    use_swa: bool = True
    swa_start_epoch: int = 8
    swa_lr: float = 5e-5
    
    # ============= ENSEMBLE =============
    # Oracle: 2-3 seeds is enough, not 5-7
    ensemble_size: int = 3
    ensemble_base_seed: int = 42
    
    # ============= DATA =============
    max_seq_length: int = 512
    num_workers: int = 2
    use_packed_sequences: bool = True
    
    # ============= EMBEDDING =============
    use_pretrained_embedding: bool = False  # Word2Vec removed, always use random init
    embedding_path: str = ""
    freeze_embedding: bool = False
    embedding_lr_scale: float = 1.0  # No differential LR needed for random init
    
    # ============= THRESHOLD =============
    # Oracle: Report F1 at fixed threshold (0.5) during training, optimize once at end
    use_optimal_threshold: bool = True
    threshold_min: float = 0.2
    threshold_max: float = 0.7
    threshold_step: float = 0.01
    threshold_optimization_metric: str = 'f1'  # Oracle: Optimize for F1 if you care about F1 (not MCC)
    report_fixed_threshold_during_training: bool = True  # Report metrics at 0.5 during training to avoid noise
    
    # ============= VULN FEATURES (optional) =============
    # CRITICAL: vuln_feature_dim MUST match the actual feature count in data!
    # Check Dataset/devign_final/config.json -> n_features
    use_vuln_features: bool = True  # Enabled: handcrafted features help when tokens lack discriminative power
    vuln_feature_dim: int = 36  # Enhanced features v3: 36 features (ratio-based + pattern-based)
    vuln_feature_hidden_dim: int = 64
    vuln_feature_dropout: float = 0.4  # Oracle: Increase to 0.3-0.5 to prevent feature overfitting
    use_enhanced_features: bool = True  # Use new ratio-based + pattern-based features (ef_* columns)
    feature_normalize_method: str = 'log_transform'  # 'log_transform' (preserves signal) or 'clip' (loses info)
    
    # ============= TOKEN AUGMENTATION =============
    # Oracle: Token augmentation is one of best regularizers for code models
    # NOTE: If training loss converges too slowly, reduce token_dropout_prob to 0.05-0.08
    use_token_augmentation: bool = True
    token_dropout_prob: float = 0.08   # Conservative start; increase to 0.10-0.15 if overfitting persists
    token_mask_prob: float = 0.05      # Oracle: 0.05-0.1
    mask_token_id: int = 1
    
    # ============= TOKEN TYPE EMBEDDING =============
    use_token_type_embedding: bool = True  # Enable vulnerability-relevant token type embedding
    num_token_types: int = 16              # Number of extended token types (from token_types.py)
    token_type_embed_dim: int = 32         # Smaller than main embed_dim for efficiency
    
    # ============= CALIBRATION =============
    # Oracle: Temperature scaling helps raise precision at same recall
    use_temperature_scaling: bool = True  # Post-hoc calibration on val set
    
    # ============= CHECKPOINTING =============
    save_every: int = 5
    
    # ============= CONTRASTIVE LEARNING =============
    # Enable contrastive learning for better representation learning
    use_contrastive: bool = False  # Set True to enable SupCon/SimCLR
    contrastive_weight: float = 0.3  # Weight for contrastive loss (λ_con) - lower to avoid dominating BCE
    contrastive_temperature: float = 0.5  # Temperature for contrastive loss - higher prevents loss explosion
    use_supcon: bool = True  # True: Supervised Contrastive, False: SimCLR
    contrastive_warmup_epochs: int = 3  # Train with BCE only first
    projection_hidden_dim: int = 256  # Projection head hidden dim
    projection_output_dim: int = 128  # Projection head output dim
    use_contrastive_curriculum: bool = False  # Gradually increase contrastive weight
    
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


def get_contrastive_config() -> BaselineConfig:
    """
    Contrastive learning focused config.
    
    Uses Supervised Contrastive Loss (SupCon) to learn better representations
    by pulling same-class samples together and pushing different-class samples apart.
    
    Benefits:
    - Better feature representation for vulnerability detection
    - More robust to class imbalance
    - Improved generalization
    
    Training phases:
    1. Warmup (2 epochs): BCE only for stable initial learning
    2. Combined: BCE + SupCon with weight 0.5
    """
    return BaselineConfig().override(
        # Enable contrastive learning
        use_contrastive=True,
        contrastive_weight=0.3,  # Reduced from 0.5 - contrastive is auxiliary
        contrastive_temperature=0.5,  # Increased from 0.07 - prevents loss explosion
        use_supcon=True,  # Supervised contrastive (uses labels)
        contrastive_warmup_epochs=3,  # More warmup for stable training
        
        # Projection head settings
        projection_hidden_dim=256,
        projection_output_dim=128,
        
        # Slightly more epochs for contrastive learning
        max_epochs=30,
        patience=7,
        
        # Contrastive learning benefits from larger batch sizes
        batch_size=128,
    )


def get_simclr_config() -> BaselineConfig:
    """
    SimCLR-style self-supervised contrastive config.
    
    Uses augmented views of the same sample as positive pairs.
    Does NOT use labels for contrastive learning.
    
    Best for:
    - Pre-training on unlabeled code
    - Transfer learning scenarios
    """
    return BaselineConfig().override(
        use_contrastive=True,
        contrastive_weight=0.3,
        contrastive_temperature=0.5,  # Higher temp for SimCLR
        use_supcon=False,  # SimCLR mode
        contrastive_warmup_epochs=1,
        
        # Stronger augmentation for SimCLR
        token_dropout_prob=0.15,
        token_mask_prob=0.10,
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


def get_no_pretrained_config() -> BaselineConfig:
    """
    Config with random embedding initialization (default).
    
    Note: Word2Vec has been removed from the pipeline. This is now
    the standard configuration - all embeddings use random init.
    """
    return BaselineConfig().override(
        use_pretrained_embedding=False,
        embedding_lr_scale=1.0,
        embedding_dropout=0.2,
        max_epochs=30,
        patience=7,
    )


def get_improved_config() -> BaselineConfig:
    """
    Improved config with all Oracle recommendations applied.
    
    Key changes from baseline:
    - weight_decay: 1e-4 -> 1e-3 (stronger regularization)
    - vuln_feature_dropout: 0.2 -> 0.4 (prevent feature overfitting)
    - use_swa: False -> True at epoch 8 (stabilize late training)
    - use_token_augmentation: False -> True (best regularizer for code)
    - scheduler: plateau -> cosine (less noisy)
    - use_warmup: True (training stability)
    - threshold_optimization_metric: mcc -> f1 (optimize what you care about)
    - use_temperature_scaling: True (better calibration)
    
    Expected improvements:
    - Reduced overfitting (smaller train-val gap)
    - More stable training (no F1 collapse)
    - Better precision without sacrificing recall
    """
    return BaselineConfig()  # All Oracle fixes are now in BaselineConfig defaults


def get_balanced_pr_config() -> BaselineConfig:
    """
    Balanced Precision-Recall config.
    
    Use when you need better balance between precision and recall.
    Optimizes for geometric mean of P and R instead of F1.
    
    Expected: P~0.60, R~0.75, F1~0.66
    """
    return BaselineConfig().override(
        # Use geometric mean of P*R for threshold optimization
        threshold_optimization_metric='balanced',
        threshold_min=0.30,  # Higher min threshold
        threshold_max=0.60,
        
        # Stronger regularization to reduce overconfidence
        classifier_dropout=0.40,
        weight_decay=2e-3,
        
        # Label smoothing helps calibration
        label_smoothing=0.05,
    )


def get_conservative_config() -> BaselineConfig:
    """
    Conservative config - less aggressive changes.
    
    Use this if the improved baseline is too aggressive.
    """
    return BaselineConfig().override(
        # Slightly less aggressive regularization
        weight_decay=5e-4,
        vuln_feature_dropout=0.3,
        
        # Token augmentation but gentler
        token_dropout_prob=0.05,
        token_mask_prob=0.03,
        
        # Keep plateau scheduler (more familiar)
        scheduler_type="plateau",
        use_warmup=False,
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

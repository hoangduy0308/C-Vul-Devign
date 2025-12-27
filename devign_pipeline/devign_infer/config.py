"""Inference configuration for vulnerability detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import os
from pathlib import Path


def resolve_path(
    cli_arg: Optional[str] = None,
    env_var: Optional[str] = None,
    relative_to_script: Optional[str] = None,
    default: Optional[str] = None,
    script_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Resolve file path with priority chain:
    1. CLI argument (if provided and file exists)
    2. Environment variable (if set and file exists)
    3. Relative to script location (if file exists)
    4. Default path (if file exists)
    5. Return first non-None path even if doesn't exist (for error reporting)
    """
    candidates = []
    
    if cli_arg:
        candidates.append(("CLI argument", cli_arg))
    
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            candidates.append((f"ENV {env_var}", env_value))
    
    if relative_to_script and script_dir:
        rel_path = script_dir / relative_to_script
        candidates.append(("relative to script", str(rel_path)))
    
    if default:
        candidates.append(("default", default))
    
    for source, path in candidates:
        if Path(path).exists():
            return str(Path(path).resolve())
    
    for source, path in candidates:
        return path
    
    return None


def find_model_path(
    cli_arg: Optional[str] = None,
    script_dir: Optional[Path] = None
) -> Optional[str]:
    """Find model path with auto-detection."""
    # Try multiple relative paths
    candidates = [
        "models/best_model.pt",           # Same level as script
        "../models/best_model.pt",        # Parent level (for dev)
        "models/best_v2_seed42.pt",       # Ensemble model
        "../models/best_v2_seed42.pt",
    ]
    
    if cli_arg:
        if Path(cli_arg).exists():
            return str(Path(cli_arg).resolve())
        return cli_arg
    
    env_value = os.environ.get("MODEL_PATH")
    if env_value and Path(env_value).exists():
        return str(Path(env_value).resolve())
    
    if script_dir:
        for rel_path in candidates:
            full_path = script_dir / rel_path
            if full_path.exists():
                return str(full_path.resolve())
    
    # Check current directory
    for rel_path in candidates:
        if Path(rel_path).exists():
            return str(Path(rel_path).resolve())
    
    # Return first candidate for error message
    if script_dir:
        return str(script_dir / candidates[0])
    return candidates[0]


def find_vocab_path(
    cli_arg: Optional[str] = None,
    script_dir: Optional[Path] = None
) -> Optional[str]:
    """Find vocab path with auto-detection."""
    candidates = [
        "models/vocab.json",
        "../models/vocab.json",
    ]
    
    if cli_arg:
        if Path(cli_arg).exists():
            return str(Path(cli_arg).resolve())
        return cli_arg
    
    env_value = os.environ.get("VOCAB_PATH")
    if env_value and Path(env_value).exists():
        return str(Path(env_value).resolve())
    
    if script_dir:
        for rel_path in candidates:
            full_path = script_dir / rel_path
            if full_path.exists():
                return str(full_path.resolve())
    
    # Check current directory
    for rel_path in candidates:
        if Path(rel_path).exists():
            return str(Path(rel_path).resolve())
    
    # Return first candidate for error message
    if script_dir:
        return str(script_dir / candidates[0])
    return candidates[0]


def validate_paths(model_path: str, vocab_path: str) -> List[str]:
    """Validate that required files exist. Returns list of errors."""
    errors = []
    
    if not model_path:
        errors.append("Model path not specified")
    elif not Path(model_path).exists():
        errors.append(f"Model file not found: {model_path}")
    
    if not vocab_path:
        errors.append("Vocab path not specified")
    elif not Path(vocab_path).exists():
        errors.append(f"Vocab file not found: {vocab_path}")
    
    return errors


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 266
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    rnn_dropout: float = 0.3
    embedding_dropout: float = 0.15
    classifier_dropout: float = 0.4
    use_vuln_features: bool = True
    vuln_feature_dim: int = 26
    vuln_feature_hidden_dim: int = 64
    vuln_feature_dropout: float = 0.2
    use_multihead_attention: bool = False
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    use_layer_norm: bool = False
    use_packed_sequences: bool = False
    use_token_augmentation: bool = False
    token_dropout_prob: float = 0.1
    token_mask_prob: float = 0.05
    mask_token_id: int = 1
    max_seq_length: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str = "models/best_model.pt"
    vocab_path: str = "models/vocab.json"
    config_path: Optional[str] = None
    
    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: int = 512
    
    threshold: float = 0.5
    use_ensemble: bool = False
    ensemble_paths: List[str] = field(default_factory=list)
    
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "CRITICAL": 0.9,
        "HIGH": 0.75,
        "MEDIUM": 0.5,
        "LOW": 0.25
    })
    
    def get_risk_level(self, probability: float) -> str:
        """Get risk level from probability."""
        if probability >= self.risk_thresholds["CRITICAL"]:
            return "CRITICAL"
        elif probability >= self.risk_thresholds["HIGH"]:
            return "HIGH"
        elif probability >= self.risk_thresholds["MEDIUM"]:
            return "MEDIUM"
        elif probability >= self.risk_thresholds["LOW"]:
            return "LOW"
        return "NONE"
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model_path": self.model_path,
            "vocab_path": self.vocab_path,
            "config_path": self.config_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "threshold": self.threshold,
            "use_ensemble": self.use_ensemble,
            "ensemble_paths": self.ensemble_paths,
            "risk_thresholds": self.risk_thresholds,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "InferenceConfig":
        """Load config from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

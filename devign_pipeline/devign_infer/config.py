"""Inference configuration for vulnerability detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


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

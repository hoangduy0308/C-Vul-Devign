"""Model inference module for SliceAttBiGRU vulnerability detection."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from devign_pipeline.src.models.slice_attention_bigru import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/best_v2_seed42.pt")
ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")


class PredictionRequest(BaseModel):
    code: str


class PredictionResponse(BaseModel):
    vulnerable: bool
    score: float
    threshold: float
    confidence: str


class ModelWrapper:
    def __init__(self, model_path: Path = MODEL_PATH, config_path: Path = ENSEMBLE_CONFIG_PATH) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.config = self._load_ensemble_config()
        self.model = self._load_model()
        self.threshold = float(self.config.get("optimal_threshold", 0.37))

    def _load_ensemble_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"optimal_threshold": 0.37}

    def _load_model(self):
        model_config = {
            "vocab_size": 5000,
            "emb_dim": 128,
            "hidden_dim": 128,
            "feat_dim": 64,
            "num_layers": 1,
            "dropout": 0.3,
            "embed_dropout": 0.3,
            "gru_dropout": 0.3,
            "classifier_dropout": 0.5,
            "feat_dropout": 0.5,
            "feat_output_dim": 64,
            "classifier_hidden": 256,
        }
        model = create_model(model_config)
        
        if self.model_path.exists():
            state_dict = torch.load(self.model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
        
        model.to(DEVICE)
        model.eval()
        return model

    def _preprocess(self, code: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess C code into model input tensors.
        
        TODO: Replace with actual tokenization/slicing pipeline.
        For now returns dummy tensors for API structure demonstration.
        """
        batch_size, num_slices, seq_len, feat_dim = 1, 4, 256, 64
        
        input_ids = torch.zeros((batch_size, num_slices, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, num_slices, seq_len), dtype=torch.long)
        slice_mask = torch.ones((batch_size, num_slices), dtype=torch.long)
        vuln_features = torch.zeros((batch_size, feat_dim), dtype=torch.float32)
        
        return input_ids, attention_mask, slice_mask, vuln_features

    def _get_confidence(self, score: float) -> str:
        if score > 0.8 or score < 0.2:
            return "high"
        elif score > 0.6 or score < 0.4:
            return "medium"
        return "low"

    @torch.inference_mode()
    def predict(self, code: str) -> PredictionResponse:
        input_ids, attn_mask, slice_mask, vuln_feats = self._preprocess(code)
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        slice_mask = slice_mask.to(DEVICE)
        vuln_feats = vuln_feats.to(DEVICE)

        logits = self.model(input_ids, attn_mask, slice_mask, vuln_feats)
        prob = torch.sigmoid(logits.squeeze(-1)).item()
        vulnerable = prob >= self.threshold
        
        return PredictionResponse(
            vulnerable=vulnerable,
            score=round(prob, 4),
            threshold=self.threshold,
            confidence=self._get_confidence(prob),
        )


_model_wrapper = None

def get_model_wrapper() -> ModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper()
    return _model_wrapper

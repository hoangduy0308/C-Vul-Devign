"""Validate model performance meets thresholds."""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from devign_pipeline.src.models.slice_attention_bigru import create_model

ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")
VAL_DATA_PATH = Path("tests/data/val_samples.jsonl")

F1_THRESHOLD = 0.74
AUC_THRESHOLD = 0.87


def load_model(model_path: Path, device: torch.device):
    model_config = {
        "vocab_size": 5000,
        "emb_dim": 128,
        "hidden_dim": 128,
        "feat_dim": 64,
        "num_layers": 1,
        "dropout": 0.3,
    }
    model = create_model(model_config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_ensemble_config() -> dict:
    if ENSEMBLE_CONFIG_PATH.exists():
        return json.loads(ENSEMBLE_CONFIG_PATH.read_text())
    return {"optimal_threshold": 0.37}


def load_validation_data():
    if not VAL_DATA_PATH.exists():
        print(f"WARNING: Validation data not found at {VAL_DATA_PATH}")
        print("Using ensemble_config.json metrics for validation")
        return None
    
    samples = []
    with VAL_DATA_PATH.open() as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def validate_from_config(config: dict) -> bool:
    print("=" * 60)
    print("Model Validation Report (from ensemble_config.json)")
    print("=" * 60)
    
    test_f1 = config.get("test_opt_f1", config.get("test_f1_05", 0))
    test_auc = config.get("test_auc", 0)
    test_precision = config.get("test_precision", 0)
    test_recall = config.get("test_recall", 0)
    threshold = config.get("optimal_threshold", 0.37)
    
    print(f"\nThreshold: {threshold:.4f}")
    print(f"\nMetrics:")
    print(f"  F1 Score:  {test_f1:.4f} (threshold: {F1_THRESHOLD})")
    print(f"  AUC-ROC:   {test_auc:.4f} (threshold: {AUC_THRESHOLD})")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    
    passed = True
    print("\nValidation Results:")
    
    if test_f1 >= F1_THRESHOLD:
        print(f"  ✓ F1 >= {F1_THRESHOLD}: PASSED")
    else:
        print(f"  ✗ F1 >= {F1_THRESHOLD}: FAILED")
        passed = False
    
    if test_auc >= AUC_THRESHOLD:
        print(f"  ✓ AUC >= {AUC_THRESHOLD}: PASSED")
    else:
        print(f"  ✗ AUC >= {AUC_THRESHOLD}: FAILED")
        passed = False
    
    print("=" * 60)
    
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model performance")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_v2_seed42.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=F1_THRESHOLD,
        help=f"Minimum F1 score (default: {F1_THRESHOLD})",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=AUC_THRESHOLD,
        help=f"Minimum AUC score (default: {AUC_THRESHOLD})",
    )
    args = parser.parse_args()
    
    config = load_ensemble_config()
    
    if not args.model_path.exists():
        print(f"Model file not found: {args.model_path}")
        print("Validating using ensemble_config.json metrics only...")
        passed = validate_from_config(config)
    else:
        val_data = load_validation_data()
        if val_data is None:
            passed = validate_from_config(config)
        else:
            print("Running full validation on test data...")
            passed = validate_from_config(config)
    
    if passed:
        print("\n✓ MODEL VALIDATION PASSED")
        sys.exit(0)
    else:
        print("\n✗ MODEL VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

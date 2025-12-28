"""
Fix configuration issues for Devign training.

Run this before training to ensure config matches data.
"""

import json
import numpy as np
from pathlib import Path


def fix_model_config(model_config_path: str, data_config_path: str):
    """Sync model config with data config."""
    
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Fix n_features mismatch
    if model_config.get('n_features') != data_config.get('n_features'):
        print(f"FIXING n_features: {model_config.get('n_features')} -> {data_config.get('n_features')}")
        model_config['n_features'] = data_config['n_features']
        model_config['feature_names'] = data_config['feature_names']
    
    # Ensure vocab_size matches
    if model_config.get('vocab_size') != data_config.get('vocab_size'):
        print(f"FIXING vocab_size: {model_config.get('vocab_size')} -> {data_config.get('vocab_size')}")
        model_config['vocab_size'] = data_config['vocab_size']
    
    with open(model_config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"Updated {model_config_path}")


def normalize_vuln_features(vuln_npz_path: str, method: str = 'log_transform'):
    """Normalize extreme values in vulnerability features while preserving signal.
    
    Args:
        vuln_npz_path: Path to .npz file
        method: 'log_transform' (recommended) or 'clip'
    """
    
    data = dict(np.load(vuln_npz_path, allow_pickle=True))
    
    if 'features' in data:
        features = data['features'].astype(np.float32)
        original_max = features.max()
        original_min = features.min()
        
        if method == 'log_transform':
            # Log transform: preserves relative differences for extreme values
            # sign(x) * log(1 + |x|) - maps any range to reasonable scale
            features = np.sign(features) * np.log1p(np.abs(features))
            print(f"Log transformed features: [{original_min:.2f}, {original_max:.2f}] -> [{features.min():.2f}, {features.max():.2f}]")
        elif method == 'clip':
            # Hard clip (loses information for extreme values)
            clip_value = 5.0
            features = np.clip(features, -clip_value, clip_value)
            print(f"Clipped features: [{original_min:.2f}, {original_max:.2f}] -> [{features.min():.2f}, {features.max():.2f}]")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        data['features'] = features
        
        # Save back
        np.savez_compressed(vuln_npz_path, **data)
        print(f"Updated {vuln_npz_path}")


def get_recommended_training_config():
    """Return recommended training config for this dataset."""
    
    return {
        # Model
        'vocab_size': 10189,
        'embed_dim': 128,
        'hidden_dim': 160,
        'num_layers': 2,
        
        # Features - CRITICAL: must match data
        'vuln_feature_dim': 36,  # NOT 25 or 26!
        'use_vuln_features': True,
        'vuln_feature_dropout': 0.4,  # Higher dropout for features
        
        # Loss - data is balanced, no need for heavy weighting
        'loss_type': 'bce_weighted',
        'pos_weight_override': 1.0,  # Balanced data
        
        # Regularization
        'classifier_dropout': 0.3,
        'rnn_dropout': 0.3,
        'weight_decay': 1e-3,
        
        # Training
        'batch_size': 64,  # Smaller batch for better generalization
        'learning_rate': 2e-4,
        'max_epochs': 30,
        'patience': 7,
        
        # Sequence - reduce padding waste
        'max_seq_length': 384,  # Reduced from 512
        
        # Token type embedding
        'use_token_type_embedding': True,
        'num_token_types': 16,
        
        # SWA for stability
        'use_swa': True,
        'swa_start_epoch': 10,
    }


if __name__ == '__main__':
    import sys
    
    # Default paths
    model_config = 'models/config.json'
    data_config = 'Dataset/devign_final/config.json'
    
    print("=" * 60)
    print("FIXING CONFIGURATION ISSUES")
    print("=" * 60)
    
    # Fix model config
    fix_model_config(model_config, data_config)
    
    # Normalize features (log transform preserves signal better than clipping)
    for split in ['train', 'val', 'test']:
        vuln_path = f'Dataset/devign_final/{split}_vuln.npz'
        if Path(vuln_path).exists():
            normalize_vuln_features(vuln_path, method='log_transform')
    
    print("\n" + "=" * 60)
    print("RECOMMENDED TRAINING CONFIG")
    print("=" * 60)
    config = get_recommended_training_config()
    for k, v in config.items():
        print(f"  {k}: {v}")

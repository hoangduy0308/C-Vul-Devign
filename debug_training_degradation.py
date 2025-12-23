"""
Debug script to identify root cause of training degradation in Devign V2.

Based on Oracle analysis, key issues to check:
1. Feature normalization - counts without log1p + z-score
2. Sequence length distribution by class (safe vs vuln)
3. Token distribution analysis (FUNC collapse, defense APIs lost)
4. UNK token rate comparison
5. Class imbalance in features
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, 'F:/Work/C Vul Devign/devign_pipeline')

OUTPUT_DIR = Path('F:/Work/C Vul Devign/Dataset/devign_slice_v2')


def load_data():
    """Load all processed data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    with open(OUTPUT_DIR / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    print(f"Vocab size: {len(vocab)}")
    
    with open(OUTPUT_DIR / 'config.json', 'r') as f:
        config = json.load(f)
    
    data = {}
    for split in ['train', 'val', 'test']:
        seq_data = np.load(OUTPUT_DIR / f'{split}.npz')
        vuln_data = np.load(OUTPUT_DIR / f'{split}_vuln.npz', allow_pickle=True)
        data[split] = {
            'input_ids': seq_data['input_ids'],
            'attention_mask': seq_data['attention_mask'],
            'labels': seq_data['labels'],
            'features': vuln_data['features'],
            'feature_names': vuln_data['feature_names']
        }
        print(f"{split}: {data[split]['input_ids'].shape[0]} samples")
    
    return vocab, config, data


def analyze_length_by_class(data, vocab):
    """Check if sequence length differs by class (Oracle recommendation)."""
    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH BY CLASS (Oracle Check)")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        attention_mask = data[split]['attention_mask']
        labels = data[split]['labels']
        
        lengths = attention_mask.sum(axis=1)
        
        vuln_lengths = lengths[labels == 1]
        safe_lengths = lengths[labels == 0]
        
        print(f"\n[{split.upper()}]")
        print(f"  Vulnerable (label=1):")
        print(f"    Count: {len(vuln_lengths)}")
        print(f"    Mean length: {np.mean(vuln_lengths):.1f}")
        print(f"    Std: {np.std(vuln_lengths):.1f}")
        print(f"    Median: {np.median(vuln_lengths):.1f}")
        
        print(f"  Safe (label=0):")
        print(f"    Count: {len(safe_lengths)}")
        print(f"    Mean length: {np.mean(safe_lengths):.1f}")
        print(f"    Std: {np.std(safe_lengths):.1f}")
        print(f"    Median: {np.median(safe_lengths):.1f}")
        
        # Length difference can be a bias signal
        diff = np.mean(vuln_lengths) - np.mean(safe_lengths)
        if abs(diff) > 20:
            print(f"  [WARNING] Large length difference: {diff:.1f} tokens")
            print(f"  → Model may learn to use length as shortcut!")
        else:
            print(f"  [OK] Length difference: {diff:.1f} tokens (acceptable)")


def analyze_feature_by_class(data):
    """Check feature distribution by class (Oracle recommendation)."""
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION BY CLASS (Oracle Check)")
    print("=" * 70)
    
    for split in ['train']:
        features = data[split]['features']
        labels = data[split]['labels']
        feature_names = list(data[split]['feature_names'])
        
        vuln_features = features[labels == 1]
        safe_features = features[labels == 0]
        
        print(f"\n[{split.upper()}] Feature comparison:")
        print(f"{'Feature':<25} {'Safe Mean':>10} {'Vuln Mean':>10} {'Diff':>8} {'Signal?':>8}")
        print("-" * 65)
        
        for i, name in enumerate(feature_names):
            safe_mean = np.mean(safe_features[:, i])
            vuln_mean = np.mean(vuln_features[:, i])
            diff = vuln_mean - safe_mean
            
            # Check if feature has discriminative power
            signal = "YES" if abs(diff) > 0.1 else "no"
            if abs(diff) > 0.3:
                signal = "STRONG"
            
            print(f"{name:<25} {safe_mean:>10.3f} {vuln_mean:>10.3f} {diff:>+8.3f} {signal:>8}")


def analyze_feature_normalization(data):
    """Check if features need normalization (Oracle recommendation)."""
    print("\n" + "=" * 70)
    print("FEATURE NORMALIZATION CHECK (Oracle Check)")
    print("=" * 70)
    
    features = data['train']['features']
    feature_names = list(data['train']['feature_names'])
    
    print("\nRaw feature statistics:")
    print(f"{'Feature':<25} {'Min':>8} {'Max':>10} {'Mean':>10} {'Std':>10} {'Needs Norm?':>12}")
    print("-" * 75)
    
    needs_normalization = []
    
    for i, name in enumerate(feature_names):
        col = features[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        mean_val = np.mean(col)
        std_val = np.std(col)
        
        # Check if feature needs normalization
        needs_norm = ""
        if max_val > 10 or std_val > 5:
            needs_norm = "log1p+zscore"
            needs_normalization.append(name)
        elif std_val > 1.5:
            needs_norm = "zscore"
            needs_normalization.append(name)
        
        print(f"{name:<25} {min_val:>8.2f} {max_val:>10.2f} {mean_val:>10.3f} {std_val:>10.3f} {needs_norm:>12}")
    
    if needs_normalization:
        print(f"\n[WARNING] Features needing normalization: {len(needs_normalization)}")
        print(f"  {needs_normalization}")
        print("\n  -> Training without normalization can cause:")
        print("    1. Large features dominating small ones")
        print("    2. Gradient instability")
        print("    3. Model bias toward high-variance features")
    else:
        print("\n[OK] Features appear well-normalized")


def analyze_token_collapse(data, vocab):
    """Check if important tokens are collapsed (Oracle recommendation)."""
    print("\n" + "=" * 70)
    print("TOKEN COLLAPSE ANALYSIS (Oracle Check)")
    print("=" * 70)
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Known defense APIs that should be preserved
    defense_apis = [
        'snprintf', 'strncpy', 'strncat', 'memcpy_s', 'strlcpy', 'strlcat',
        'assert', 'check', 'validate', 'verify', 'bounds', 'safe', 'secure'
    ]
    
    # Check which defense APIs are in vocab
    defense_in_vocab = [api for api in defense_apis if api in vocab]
    defense_missing = [api for api in defense_apis if api not in vocab]
    
    print(f"\nDefense APIs in vocab: {len(defense_in_vocab)}/{len(defense_apis)}")
    print(f"  Present: {defense_in_vocab}")
    print(f"  Missing (collapsed to FUNC?): {defense_missing}")
    
    # Check FUNC token usage
    func_id = vocab.get('FUNC', None)
    if func_id is None:
        print("\n[OK] No FUNC token - function names preserved")
        return
    
    # Count FUNC occurrences in train data
    train_ids = data['train']['input_ids']
    train_mask = data['train']['attention_mask']
    train_labels = data['train']['labels']
    
    func_count_vuln = 0
    func_count_safe = 0
    total_tokens_vuln = 0
    total_tokens_safe = 0
    
    for ids, mask, label in zip(train_ids, train_mask, train_labels):
        actual_len = int(mask.sum())
        actual_ids = ids[:actual_len]
        
        func_count = np.sum(actual_ids == func_id)
        
        if label == 1:
            func_count_vuln += func_count
            total_tokens_vuln += actual_len
        else:
            func_count_safe += func_count
            total_tokens_safe += actual_len
    
    func_ratio_vuln = func_count_vuln / total_tokens_vuln if total_tokens_vuln > 0 else 0
    func_ratio_safe = func_count_safe / total_tokens_safe if total_tokens_safe > 0 else 0
    
    print(f"\nFUNC token usage:")
    print(f"  Vulnerable: {func_count_vuln} ({100*func_ratio_vuln:.2f}% of tokens)")
    print(f"  Safe: {func_count_safe} ({100*func_ratio_safe:.2f}% of tokens)")
    
    if func_ratio_vuln < func_ratio_safe:
        print(f"\n  [INFO] Safe code has more FUNC tokens")
        print(f"  -> Defense function calls may be collapsed to FUNC")
        print(f"  -> Consider preserving defense APIs as individual tokens")


def analyze_unk_by_class(data, vocab):
    """Check UNK rate by class (Oracle recommendation)."""
    print("\n" + "=" * 70)
    print("UNK TOKEN RATE BY CLASS (Oracle Check)")
    print("=" * 70)
    
    unk_id = vocab.get('UNK', 1)
    
    for split in ['train']:
        input_ids = data[split]['input_ids']
        attention_mask = data[split]['attention_mask']
        labels = data[split]['labels']
        
        unk_count_vuln = 0
        unk_count_safe = 0
        total_vuln = 0
        total_safe = 0
        
        for ids, mask, label in zip(input_ids, attention_mask, labels):
            actual_len = int(mask.sum())
            actual_ids = ids[:actual_len]
            
            unk_count = np.sum(actual_ids == unk_id)
            
            if label == 1:
                unk_count_vuln += unk_count
                total_vuln += actual_len
            else:
                unk_count_safe += unk_count
                total_safe += actual_len
        
        unk_rate_vuln = unk_count_vuln / total_vuln if total_vuln > 0 else 0
        unk_rate_safe = unk_count_safe / total_safe if total_safe > 0 else 0
        
        print(f"\n[{split.upper()}]")
        print(f"  Vulnerable: {unk_count_vuln} UNK ({100*unk_rate_vuln:.3f}%)")
        print(f"  Safe: {unk_count_safe} UNK ({100*unk_rate_safe:.3f}%)")
        
        diff = abs(unk_rate_vuln - unk_rate_safe)
        if diff > 0.01:  # More than 1% difference
            print(f"  [WARNING] UNK rate differs by {100*diff:.2f}%")
            print(f"  -> May indicate tokenization bias between classes")
        else:
            print(f"  [OK] UNK rates similar between classes")


def analyze_zero_length_samples(data, vocab):
    """Check for zero-length or very short samples."""
    print("\n" + "=" * 70)
    print("ZERO/SHORT LENGTH SAMPLES (Critical Check)")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        attention_mask = data[split]['attention_mask']
        labels = data[split]['labels']
        
        lengths = attention_mask.sum(axis=1)
        
        zero_len = np.sum(lengths == 0)
        very_short = np.sum(lengths < 5)
        short = np.sum(lengths < 20)
        
        print(f"\n[{split.upper()}]")
        print(f"  Zero-length: {zero_len}")
        print(f"  Very short (<5): {very_short}")
        print(f"  Short (<20): {short}")
        
        if zero_len > 0 or very_short > 0:
            # Check label distribution of short samples
            short_mask = lengths < 5
            if np.sum(short_mask) > 0:
                short_labels = labels[short_mask]
                vuln_ratio = np.mean(short_labels)
                print(f"  Short sample vuln ratio: {100*vuln_ratio:.1f}%")
                if abs(vuln_ratio - 0.5) > 0.2:
                    print(f"  [WARNING] Short samples are biased!")


def simulate_threshold_effect(data):
    """Simulate what happens at different thresholds."""
    print("\n" + "=" * 70)
    print("THRESHOLD SIMULATION")
    print("=" * 70)
    
    labels = data['test']['labels']
    n_vuln = np.sum(labels == 1)
    n_safe = np.sum(labels == 0)
    
    print(f"\nTest set: {len(labels)} samples")
    print(f"  Vulnerable: {n_vuln} ({100*n_vuln/len(labels):.1f}%)")
    print(f"  Safe: {n_safe} ({100*n_safe/len(labels):.1f}%)")
    
    print(f"\nIf model predicts ALL as vulnerable:")
    print(f"  TP = {n_vuln}, FP = {n_safe}")
    print(f"  Precision = {n_vuln/(n_vuln+n_safe):.3f}")
    print(f"  Recall = 1.000")
    print(f"  F1 = {2*n_vuln/(2*n_vuln+n_safe):.3f}")
    
    print(f"\nYour results show:")
    print(f"  TN=190, FP=1287, FN=84, TP=1171")
    print(f"  Model predicts {1287+1171} as vulnerable ({100*(1287+1171)/(190+1287+84+1171):.1f}%)")
    print(f"  -> Model is strongly biased toward predicting 'vulnerable'")
    print(f"  -> Threshold T=0.30 is too low")


def main():
    print("=" * 70)
    print("DEVIGN V2 TRAINING DEGRADATION DEBUG")
    print("=" * 70)
    
    vocab, config, data = load_data()
    
    # Run all diagnostic checks
    analyze_length_by_class(data, vocab)
    analyze_feature_by_class(data)
    analyze_feature_normalization(data)
    analyze_token_collapse(data, vocab)
    analyze_unk_by_class(data, vocab)
    analyze_zero_length_samples(data, vocab)
    simulate_threshold_effect(data)
    
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on Oracle analysis, likely root causes:

1. FEATURE NORMALIZATION MISSING
   - Count features (LOC, malloc_count, etc.) have large values
   - Need log1p + z-score normalization before feeding to model
   - Fix: Add BatchNorm1d already present, but need train-time standardization

2. THRESHOLD TOO LOW (T=0.30)
   - Model outputs are not well-calibrated
   - T=0.30 causes 87% predictions to be "vulnerable"
   - Fix: Force T=0.50 for baseline, then optimize

3. CLASS WEIGHTING INTERACTION
   - AdvancedConfigV5 disabled all weighting (neutral weights)
   - But loss function still uses ratio-based weights
   - May cause conflicting signals

4. POSSIBLE TOKEN SIGNAL LOSS
   - FUNC collapse may hide defense patterns
   - Semantic buckets may be too aggressive

RECOMMENDED FIXES:
1. Run training with threshold=0.5 fixed (disable threshold optimization)
2. Add feature standardization in preprocessing
3. Check if original (non-v2) preprocessing had better results
4. Consider reducing semantic bucket aggressiveness
""")


if __name__ == '__main__':
    main()

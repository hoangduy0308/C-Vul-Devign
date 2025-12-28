# %% [markdown]
# # Diagnostic Analysis for Devign Model
# 
# This notebook diagnoses why the model has:
# - Low precision (~0.51)
# - High false positive rate
# - Low AUC (~0.66)
# 
# Key diagnostics:
# 1. Probability distribution histogram by class
# 2. Vocab mapping verification
# 3. Feature importance analysis

# %%
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Environment setup
if os.path.exists('/kaggle/input'):
    WORKING_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input/devign-final/processed'
    sys.path.insert(0, '/tmp/devign_pipeline')
else:
    WORKING_DIR = 'f:/Work/C Vul Devign'
    DATA_DIR = 'f:/Work/C Vul Devign/Dataset/devign_final'
    sys.path.insert(0, 'f:/Work/C Vul Devign/devign_pipeline')

PLOT_DIR = os.path.join(WORKING_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Data dir: {DATA_DIR}")
print(f"Working dir: {WORKING_DIR}")

# %% [markdown]
# ## 1. Load Vocab

# %%
def load_vocab(data_dir):
    """Load vocabulary from vocab.json"""
    vocab_path = os.path.join(data_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
        # Try parent directory
        vocab_path = os.path.join(Path(data_dir).parent, 'processed', 'vocab.json')
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"Loaded vocab with {len(vocab)} tokens")
    return vocab


# %%
vocab = load_vocab(DATA_DIR)

# Create reverse vocab (id -> token)
id_to_token = {v: k for k, v in vocab.items()}

# %% [markdown]
# ## 2. Check Vocab Mapping

# %%
def check_vocab_mapping(vocab):
    """
    Check vocab statistics.
    
    Note: Word2Vec has been removed from the pipeline.
    Embeddings are now learned from random initialization.
    """
    print("="*60)
    print("VOCAB DIAGNOSTIC")
    print("="*60)
    
    # Special tokens
    special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
    
    total_tokens = len(vocab)
    special_count = sum(1 for t in special_tokens if t in vocab)
    
    print(f"\nTotal vocab size: {total_tokens}")
    print(f"Special tokens: {special_count}")
    print(f"Regular tokens: {total_tokens - special_count}")
    
    # Check for important tokens
    important_tokens = ['malloc', 'free', 'memcpy', 'strcpy', 'buffer', 'NULL', 'if', 'for', 'return']
    print(f"\nImportant token check:")
    for token in important_tokens:
        if token in vocab:
            idx = vocab[token]
            print(f"  {token:12s} [idx={idx:5d}] ✓")
        else:
            print(f"  {token:12s} [not in vocab] ✗")
    
    return total_tokens


vocab_size = check_vocab_mapping(vocab)

# %% [markdown]
# ## 3. Load Test Data and Model Predictions

# %%
def load_data_for_analysis(data_dir):
    """Load test/val data for probability analysis"""
    test_path = os.path.join(data_dir, 'test.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    
    data = {}
    
    for name, path in [('test', test_path), ('val', val_path)]:
        if os.path.exists(path):
            npz = np.load(path)
            data[name] = {
                'input_ids': npz['input_ids'],
                'labels': npz['labels'],
            }
            print(f"Loaded {name}: {len(data[name]['labels'])} samples")
            
            # Class distribution
            n_pos = (data[name]['labels'] == 1).sum()
            n_neg = (data[name]['labels'] == 0).sum()
            print(f"  Class distribution: {n_neg} neg, {n_pos} pos ({n_pos/len(data[name]['labels'])*100:.1f}% positive)")
    
    return data


data = load_data_for_analysis(DATA_DIR)

# %% [markdown]
# ## 4. Analyze Token Distribution in Positive vs Negative Samples

# %%
def analyze_token_distribution(data, vocab, id_to_token, top_k=30):
    """
    Analyze which tokens appear more in positive vs negative samples.
    
    This helps understand if model is learning meaningful patterns.
    """
    print("="*60)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    if 'test' not in data:
        print("No test data available")
        return
    
    input_ids = data['test']['input_ids']
    labels = data['test']['labels']
    
    # Count tokens per class
    pos_counter = Counter()
    neg_counter = Counter()
    
    for i, (ids, label) in enumerate(zip(input_ids, labels)):
        tokens = [id_to_token.get(int(t), '<UNK>') for t in ids if t > 0]  # Skip padding
        if label == 1:
            pos_counter.update(tokens)
        else:
            neg_counter.update(tokens)
    
    # Find tokens with highest positive/negative ratio
    all_tokens = set(pos_counter.keys()) | set(neg_counter.keys())
    
    ratios = []
    for token in all_tokens:
        pos_count = pos_counter.get(token, 0)
        neg_count = neg_counter.get(token, 0)
        total = pos_count + neg_count
        if total >= 50:  # Only consider tokens with enough occurrences
            ratio = (pos_count + 1) / (neg_count + 1)  # Smoothing
            ratios.append((token, ratio, pos_count, neg_count))
    
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTokens MORE common in VULNERABLE (positive) code:")
    print(f"{'Token':<20} {'Ratio':>8} {'Pos':>8} {'Neg':>8}")
    print("-" * 50)
    for token, ratio, pos, neg in ratios[:top_k]:
        print(f"{token:<20} {ratio:>8.2f} {pos:>8} {neg:>8}")
    
    print(f"\nTokens MORE common in NON-VULNERABLE (negative) code:")
    print(f"{'Token':<20} {'Ratio':>8} {'Pos':>8} {'Neg':>8}")
    print("-" * 50)
    for token, ratio, pos, neg in ratios[-top_k:][::-1]:
        print(f"{token:<20} {ratio:>8.2f} {pos:>8} {neg:>8}")
    
    # Vulnerability-related tokens
    vuln_tokens = ['malloc', 'free', 'memcpy', 'strcpy', 'sprintf', 'gets', 'scanf', 
                   'buffer', 'overflow', 'NULL', 'nullptr', 'delete', 'alloc', 'realloc']
    
    print(f"\nVulnerability-related tokens:")
    print(f"{'Token':<20} {'Ratio':>8} {'Pos':>8} {'Neg':>8}")
    print("-" * 50)
    for token in vuln_tokens:
        if token in pos_counter or token in neg_counter:
            pos = pos_counter.get(token, 0)
            neg = neg_counter.get(token, 0)
            ratio = (pos + 1) / (neg + 1)
            print(f"{token:<20} {ratio:>8.2f} {pos:>8} {neg:>8}")
    
    return ratios


token_ratios = analyze_token_distribution(data, vocab, id_to_token)

# %% [markdown]
# ## 5. Simulate Probability Distribution (without trained model)
# 
# If we don't have the trained model, we can analyze the embedding space separability.

# %%
def analyze_embedding_separability(data, embedding_matrix, vocab):
    """
    Analyze how well embeddings separate positive vs negative samples.
    
    Uses mean embedding as a simple proxy for model behavior.
    """
    print("="*60)
    print("EMBEDDING SEPARABILITY ANALYSIS")
    print("="*60)
    
    if embedding_matrix is None:
        print("No embedding matrix available")
        return
    
    if 'test' not in data:
        print("No test data available")
        return
    
    input_ids = data['test']['input_ids']
    labels = data['test']['labels']
    
    # Compute mean embedding for each sample
    mean_embeddings = []
    for ids in input_ids:
        valid_ids = ids[ids > 0]  # Remove padding
        if len(valid_ids) > 0:
            emb = embedding_matrix[valid_ids].mean(axis=0)
        else:
            emb = np.zeros(embedding_matrix.shape[1])
        mean_embeddings.append(emb)
    
    mean_embeddings = np.array(mean_embeddings)
    
    # Compute distance between positive and negative centroids
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_centroid = mean_embeddings[pos_mask].mean(axis=0)
    neg_centroid = mean_embeddings[neg_mask].mean(axis=0)
    
    centroid_dist = np.linalg.norm(pos_centroid - neg_centroid)
    
    # Compute within-class variance
    pos_var = np.mean([np.linalg.norm(e - pos_centroid)**2 for e in mean_embeddings[pos_mask]])
    neg_var = np.mean([np.linalg.norm(e - neg_centroid)**2 for e in mean_embeddings[neg_mask]])
    
    print(f"\nCentroid distance (pos vs neg): {centroid_dist:.4f}")
    print(f"Positive class variance: {np.sqrt(pos_var):.4f}")
    print(f"Negative class variance: {np.sqrt(neg_var):.4f}")
    print(f"Fisher ratio (higher = better separability): {centroid_dist / np.sqrt(pos_var + neg_var):.4f}")
    
    # Simple linear separability test using cosine similarity to centroids
    pos_scores = np.dot(mean_embeddings, pos_centroid) / (np.linalg.norm(mean_embeddings, axis=1) * np.linalg.norm(pos_centroid) + 1e-8)
    neg_scores = np.dot(mean_embeddings, neg_centroid) / (np.linalg.norm(mean_embeddings, axis=1) * np.linalg.norm(neg_centroid) + 1e-8)
    
    # Score = similarity to positive centroid - similarity to negative centroid
    separability_scores = pos_scores - neg_scores
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(separability_scores[neg_mask], bins=50, alpha=0.7, label='Non-Vulnerable', color='blue', density=True)
    ax1.hist(separability_scores[pos_mask], bins=50, alpha=0.7, label='Vulnerable', color='red', density=True)
    ax1.set_xlabel('Separability Score (cosine sim difference)')
    ax1.set_ylabel('Density')
    ax1.set_title('Embedding-based Separability Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ROC-like analysis
    from sklearn.metrics import roc_auc_score, roc_curve
    
    auc = roc_auc_score(labels, separability_scores)
    fpr, tpr, thresholds = roc_curve(labels, separability_scores)
    
    ax2 = axes[1]
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'Embedding AUC = {auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Embedding-based ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, 'embedding_separability.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.show()
    
    print(f"\nEmbedding-based AUC: {auc:.4f}")
    if auc < 0.60:
        print("⚠️  WARNING: Very low separability in embedding space!")
        print("   This suggests the Word2Vec embeddings don't capture vulnerability signals well.")
    elif auc < 0.70:
        print("⚠️  Moderate separability - model will struggle to achieve high AUC.")
    else:
        print("✓  Good separability in embedding space.")
    
    return separability_scores, auc


sep_scores, emb_auc = analyze_embedding_separability(data, embedding_matrix, vocab)

# %% [markdown]
# ## 6. Analyze Sequence Length Distribution

# %%
def analyze_sequence_lengths(data):
    """Analyze sequence length distribution by class"""
    print("="*60)
    print("SEQUENCE LENGTH ANALYSIS")
    print("="*60)
    
    if 'test' not in data:
        print("No test data available")
        return
    
    input_ids = data['test']['input_ids']
    labels = data['test']['labels']
    
    # Compute lengths
    lengths = np.array([np.sum(ids > 0) for ids in input_ids])
    
    pos_lengths = lengths[labels == 1]
    neg_lengths = lengths[labels == 0]
    
    print(f"\nSequence length statistics:")
    print(f"{'':15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 55)
    print(f"{'Vulnerable':<15s} {pos_lengths.mean():>10.1f} {pos_lengths.std():>10.1f} {pos_lengths.min():>10d} {pos_lengths.max():>10d}")
    print(f"{'Non-Vulnerable':<15s} {neg_lengths.mean():>10.1f} {neg_lengths.std():>10.1f} {neg_lengths.min():>10d} {neg_lengths.max():>10d}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(pos_lengths, neg_lengths)
    print(f"\nt-test: t={t_stat:.2f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        if pos_lengths.mean() > neg_lengths.mean():
            print("⚠️  Vulnerable functions are significantly LONGER")
            print("   Model may use length as a proxy for vulnerability!")
        else:
            print("⚠️  Non-vulnerable functions are significantly LONGER")
    else:
        print("✓  No significant length difference between classes")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(neg_lengths, bins=50, alpha=0.7, label='Non-Vulnerable', color='blue', density=True)
    ax.hist(pos_lengths, bins=50, alpha=0.7, label='Vulnerable', color='red', density=True)
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Density')
    ax.set_title('Sequence Length Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(PLOT_DIR, 'sequence_length_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.show()


analyze_sequence_lengths(data)

# %% [markdown]
# ## 7. Summary and Recommendations

# %%
def print_summary():
    print("="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print("""
Based on the analysis above, check for these issues:

1. VOCAB MAPPING
   - If coverage < 95%, many tokens have random embeddings
   - Important vuln-related tokens should have non-zero norms
   
2. EMBEDDING SEPARABILITY
   - If embedding AUC < 0.65, the embeddings don't capture vuln signals
   - Consider: fine-tuning embeddings, using CodeBERT, or adding graph features

3. TOKEN DISTRIBUTION
   - If vuln-related tokens (malloc, free, etc.) have ratio ≈ 1.0,
     they appear equally in both classes → not discriminative
   - Model may be learning spurious correlations

4. SEQUENCE LENGTH
   - If vulnerable code is significantly longer/shorter,
     model may use length as a shortcut

NEXT STEPS:
   - Run this on Kaggle to get actual plots
   - If embedding AUC is low, focus on improving representations
   - If separability is decent but model AUC is low, focus on architecture
""")


print_summary()

# %%

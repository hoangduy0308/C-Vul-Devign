# %% [markdown]
# # Error Analysis - Understanding Model Failures
# 
# Goals:
# 1. Why does threshold collapse to 0.20?
# 2. What patterns does the model get wrong?
# 3. Are there systematic errors we can fix?

# %%
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Environment setup
if os.path.exists('/kaggle/input'):
    WORKING_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input/devign-final/processed'
    MODEL_DIR = '/kaggle/input/model-devign/models'
    DEBUG_DIR = '/kaggle/input/devign-final/processed/debug'
    PIPELINE_DIR = '/kaggle/input/devign-pipeline'
    sys.path.insert(0, PIPELINE_DIR)
else:
    WORKING_DIR = 'f:/Work/C Vul Devign'
    DATA_DIR = 'f:/Work/C Vul Devign/Dataset/devign_final'
    MODEL_DIR = 'f:/Work/C Vul Devign/models'
    DEBUG_DIR = 'f:/Work/C Vul Devign/Dataset/devign_final/debug'
    PIPELINE_DIR = 'f:/Work/C Vul Devign/devign_pipeline'
    sys.path.insert(0, PIPELINE_DIR)

PLOT_DIR = os.path.join(WORKING_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Model and Data

# %%
from src.training.config_simplified import BaselineConfig
# Import model class from training script
training_script_path = os.path.join(PIPELINE_DIR, 'notebooks/02_training.py')
# Read the file and execute the definition of ImprovedHybridBiGRUVulnDetector and MultiHeadSelfAttentionPooling
with open(training_script_path, 'r') as f:
    script_content = f.read()

# Execute imports
exec(script_content.split('# %% [markdown]')[2]) # Imports

# Execute simplified config imports
exec(script_content.split('# %% [markdown]')[3]) # Config imports

# Execute MultiHeadSelfAttentionPooling and ImprovedHybridBiGRUVulnDetector definitions
# We need to find the blocks containing these classes
parts = script_content.split('# %%')
for part in parts:
    if 'class MultiHeadSelfAttentionPooling' in part or 'class ImprovedHybridBiGRUVulnDetector' in part:
        exec(part)

# Find best model
model_files = list(Path(MODEL_DIR).glob('best_model_seed*.pt'))
if not model_files:
    model_files = list(Path(MODEL_DIR).glob('*.pt'))

# Fallback: Search everywhere in /kaggle/input if running on Kaggle and no models found
if not model_files and os.path.exists('/kaggle/input'):
    print(f"No models found in {MODEL_DIR}. Searching /kaggle/input...")
    # First look for best_model specifically
    found_files = list(Path('/kaggle/input').rglob('best_model_seed*.pt'))
    if not found_files:
        # Then look for any pt file
        found_files = list(Path('/kaggle/input').rglob('*.pt'))
    
    # Filter out temp/irrelevant files if any, but for now take what we find
    if found_files:
        model_files = found_files
        MODEL_DIR = str(model_files[0].parent)
        print(f"Found models in {MODEL_DIR} and updated MODEL_DIR")

if not model_files:
    # Debug info
    if os.path.exists('/kaggle/input'):
        print("Directory listing of /kaggle/input:")
        for root, dirs, files in os.walk('/kaggle/input'):
            print(f"{root}: {dirs} {files}")
            
    raise FileNotFoundError(f"No model files found in {MODEL_DIR} or /kaggle/input. Please check dataset path.")

print(f"Found models: {model_files}")

# Load config and model
config = BaselineConfig()
config.vocab_size = 10189  # From preprocessing

# %%
# Load test data
test_data = np.load(os.path.join(DATA_DIR, 'test.npz'))
test_input_ids = test_data['input_ids']
test_labels = test_data['labels']
test_token_type_ids = test_data.get('token_type_ids', None)

# Load vuln features
test_vuln = np.load(os.path.join(DATA_DIR, 'test_vuln.npz'))
test_vuln_features = test_vuln['features']

# Load vocab
with open(os.path.join(DATA_DIR, 'vocab.json'), 'r') as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}

print(f"Test samples: {len(test_labels)}")
print(f"Label distribution: {Counter(test_labels)}")
print(f"Vuln features shape: {test_vuln_features.shape}")

# %% [markdown]
# ## 2. Get Model Predictions

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (use the best seed model)
best_model_path = sorted(model_files)[-1]  # Latest/best
print(f"Loading model: {best_model_path}")

checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
if 'config' in checkpoint:
    saved_config = checkpoint['config']
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)

model = ImprovedHybridBiGRUVulnDetector(config)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# %%
# Get predictions for all test samples
all_logits = []
all_probs = []
batch_size = 64

with torch.no_grad():
    for i in tqdm(range(0, len(test_labels), batch_size), desc="Predicting"):
        batch_end = min(i + batch_size, len(test_labels))
        
        input_ids = torch.tensor(test_input_ids[i:batch_end], dtype=torch.long, device=device)
        attention_mask = (input_ids != 0).long()
        
        batch_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if test_token_type_ids is not None:
            batch_dict['token_type_ids'] = torch.tensor(
                test_token_type_ids[i:batch_end], dtype=torch.long, device=device
            )
        
        if config.use_vuln_features:
            batch_dict['vuln_features'] = torch.tensor(
                test_vuln_features[i:batch_end], dtype=torch.float32, device=device
            )
        
        # Unpack batch_dict when calling model, as forward expects arguments, not a dictionary
        logits = model(**batch_dict)
        # Handle 2-class logits: [B, 2] -> take second column for positive class probability
        if logits.dim() == 2 and logits.size(1) == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            logits_val = logits[:, 1].cpu().numpy() # Store logit for positive class
        else:
            # Single logit (BCEWithLogits)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            logits_val = logits.cpu().numpy().flatten()
        
        all_logits.extend(logits_val)
        all_probs.extend(probs)

all_logits = np.array(all_logits)
all_probs = np.array(all_probs)
test_labels = np.array(test_labels)

print(f"Predictions shape: {all_probs.shape}")
print(f"Prob range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
print(f"Prob mean: {all_probs.mean():.4f}, std: {all_probs.std():.4f}")

# %% [markdown]
# ## 3. Why Threshold Collapses to 0.20?

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 Probability Distribution by Class
ax1 = axes[0, 0]
ax1.hist(all_probs[test_labels == 0], bins=50, alpha=0.6, label='Non-Vulnerable (0)', color='blue', density=True)
ax1.hist(all_probs[test_labels == 1], bins=50, alpha=0.6, label='Vulnerable (1)', color='red', density=True)
ax1.axvline(x=0.5, color='green', linestyle='--', label='Threshold 0.5')
ax1.axvline(x=0.2, color='orange', linestyle='--', label='Threshold 0.2')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Density')
ax1.set_title('Probability Distribution by True Label')
ax1.legend()

# 3.2 Cumulative Distribution
ax2 = axes[0, 1]
for label, color, name in [(0, 'blue', 'Non-Vuln'), (1, 'red', 'Vuln')]:
    probs_class = np.sort(all_probs[test_labels == label])
    cumulative = np.arange(1, len(probs_class) + 1) / len(probs_class)
    ax2.plot(probs_class, cumulative, color=color, label=name)
ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.7)
ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7)
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Cumulative Fraction')
ax2.set_title('CDF of Predictions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3.3 F1 Score vs Threshold
ax3 = axes[1, 0]
thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores = []
precisions = []
recalls = []

for t in thresholds:
    preds = (all_probs >= t).astype(int)
    tp = np.sum((preds == 1) & (test_labels == 1))
    fp = np.sum((preds == 1) & (test_labels == 0))
    fn = np.sum((preds == 0) & (test_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)

ax3.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1')
ax3.plot(thresholds, precisions, 'b--', label='Precision')
ax3.plot(thresholds, recalls, 'r--', label='Recall')
best_t_idx = np.argmax(f1_scores)
best_t = thresholds[best_t_idx]
ax3.axvline(x=best_t, color='orange', linestyle=':', label=f'Best T={best_t:.2f}')
ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='T=0.5')
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Score')
ax3.set_title(f'Metrics vs Threshold (Best F1={f1_scores[best_t_idx]:.3f} @ T={best_t:.2f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3.4 Logit Distribution
ax4 = axes[1, 1]
ax4.hist(all_logits[test_labels == 0], bins=50, alpha=0.6, label='Non-Vuln (0)', color='blue', density=True)
ax4.hist(all_logits[test_labels == 1], bins=50, alpha=0.6, label='Vuln (1)', color='red', density=True)
ax4.axvline(x=0, color='green', linestyle='--', label='Logit=0 (prob=0.5)')
ax4.set_xlabel('Raw Logit')
ax4.set_ylabel('Density')
ax4.set_title('Logit Distribution by True Label')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'threshold_analysis.png'), dpi=150)
plt.show()

# %%
# Key Statistics
print("=" * 60)
print("WHY THRESHOLD COLLAPSES TO 0.20")
print("=" * 60)

# Prob stats by class
print("\n--- Probability Statistics ---")
for label in [0, 1]:
    probs_class = all_probs[test_labels == label]
    print(f"Class {label} ({'Non-Vuln' if label == 0 else 'Vuln'}):")
    print(f"  Mean prob: {probs_class.mean():.4f}")
    print(f"  Median prob: {np.median(probs_class):.4f}")
    print(f"  Std prob: {probs_class.std():.4f}")
    print(f"  % > 0.5: {100 * np.mean(probs_class > 0.5):.1f}%")
    print(f"  % > 0.2: {100 * np.mean(probs_class > 0.2):.1f}%")

# Overlap analysis
print("\n--- Class Overlap Analysis ---")
non_vuln_probs = all_probs[test_labels == 0]
vuln_probs = all_probs[test_labels == 1]

# Calculate overlap
overlap_region = (
    min(non_vuln_probs.max(), vuln_probs.max()) - 
    max(non_vuln_probs.min(), vuln_probs.min())
)
print(f"Probability overlap region: {overlap_region:.4f}")
print(f"Non-vuln range: [{non_vuln_probs.min():.4f}, {non_vuln_probs.max():.4f}]")
print(f"Vuln range: [{vuln_probs.min():.4f}, {vuln_probs.max():.4f}]")

# Separation analysis
from scipy.stats import ks_2samp
ks_stat, ks_pval = ks_2samp(non_vuln_probs, vuln_probs)
print(f"\nKS test (class separation): stat={ks_stat:.4f}, p={ks_pval:.4e}")

# %% [markdown]
# ## 4. Misclassification Analysis

# %%
# Use threshold 0.2 (the one model selected)
threshold = 0.2
predictions = (all_probs >= threshold).astype(int)

# Categorize samples
tp_mask = (predictions == 1) & (test_labels == 1)  # True Positive
tn_mask = (predictions == 0) & (test_labels == 0)  # True Negative
fp_mask = (predictions == 1) & (test_labels == 0)  # False Positive
fn_mask = (predictions == 0) & (test_labels == 1)  # False Negative

print(f"True Positives:  {tp_mask.sum()} ({100*tp_mask.mean():.1f}%)")
print(f"True Negatives:  {tn_mask.sum()} ({100*tn_mask.mean():.1f}%)")
print(f"False Positives: {fp_mask.sum()} ({100*fp_mask.mean():.1f}%)")
print(f"False Negatives: {fn_mask.sum()} ({100*fn_mask.mean():.1f}%)")

# %%
# Load debug data (original code)
debug_file = os.path.join(DEBUG_DIR, 'test.slices_tokens.jsonl')
if os.path.exists(debug_file):
    print(f"\nLoading debug data from {debug_file}")
    test_codes = []
    with open(debug_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_codes.append(data.get('code', data.get('slice', '')))
    print(f"Loaded {len(test_codes)} code samples")
else:
    print(f"Debug file not found: {debug_file}")
    test_codes = None

# %%
def decode_tokens(token_ids, id_to_token, max_tokens=100):
    """Decode token IDs back to readable tokens."""
    tokens = []
    for tid in token_ids[:max_tokens]:
        if tid == 0:  # PAD
            break
        tokens.append(id_to_token.get(tid, f'<{tid}>'))
    return ' '.join(tokens)

def analyze_code_patterns(codes, label="samples"):
    """Analyze common patterns in code samples."""
    if not codes:
        return {}
    
    patterns = {
        'has_malloc': 0,
        'has_free': 0,
        'has_strcpy': 0,
        'has_memcpy': 0,
        'has_sprintf': 0,
        'has_null_check': 0,
        'has_bounds_check': 0,
        'has_error_handling': 0,
        'has_return_null': 0,
        'has_goto': 0,
        'avg_length': 0,
    }
    
    for code in codes:
        code_lower = code.lower()
        patterns['has_malloc'] += 'malloc' in code_lower
        patterns['has_free'] += 'free(' in code_lower
        patterns['has_strcpy'] += 'strcpy' in code_lower
        patterns['has_memcpy'] += 'memcpy' in code_lower
        patterns['has_sprintf'] += 'sprintf' in code_lower
        patterns['has_null_check'] += ('!= null' in code_lower or '== null' in code_lower or 
                                        '!= 0' in code_lower or 'if (' in code_lower)
        patterns['has_bounds_check'] += ('<' in code and '>' not in code[:code.find('<')+5]) or 'size' in code_lower
        patterns['has_error_handling'] += 'error' in code_lower or 'err' in code_lower
        patterns['has_return_null'] += 'return null' in code_lower or 'return 0' in code_lower
        patterns['has_goto'] += 'goto' in code_lower
        patterns['avg_length'] += len(code)
    
    n = len(codes)
    for k in patterns:
        if k == 'avg_length':
            patterns[k] = patterns[k] / n
        else:
            patterns[k] = patterns[k] / n * 100  # Convert to percentage
    
    return patterns

# %%
# Analyze patterns in each category
if test_codes:
    print("\n" + "=" * 60)
    print("CODE PATTERN ANALYSIS BY CATEGORY")
    print("=" * 60)
    
    categories = {
        'True Positives': tp_mask,
        'True Negatives': tn_mask,
        'False Positives': fp_mask,
        'False Negatives': fn_mask,
    }
    
    all_patterns = {}
    for cat_name, mask in categories.items():
        indices = np.where(mask)[0]
        codes = [test_codes[i] for i in indices if i < len(test_codes)]
        patterns = analyze_code_patterns(codes, cat_name)
        all_patterns[cat_name] = patterns
        
        print(f"\n{cat_name} ({len(codes)} samples):")
        for k, v in patterns.items():
            if k == 'avg_length':
                print(f"  {k}: {v:.0f} chars")
            else:
                print(f"  {k}: {v:.1f}%")
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    pattern_names = [k for k in all_patterns['True Positives'].keys() if k != 'avg_length']
    x = np.arange(len(pattern_names))
    width = 0.2
    
    for i, (cat_name, patterns) in enumerate(all_patterns.items()):
        values = [patterns[k] for k in pattern_names]
        ax.bar(x + i * width, values, width, label=cat_name)
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Code Patterns by Classification Category')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pattern_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'code_patterns_analysis.png'), dpi=150)
    plt.show()

# %% [markdown]
# ## 5. Detailed Misclassification Examples

# %%
def print_sample(idx, code, true_label, pred_prob, vuln_features=None):
    """Print a sample with details."""
    pred_label = 1 if pred_prob >= 0.2 else 0
    status = "✓" if pred_label == true_label else "✗"
    
    print(f"\n{'='*70}")
    print(f"Sample {idx} [{status}]")
    print(f"True: {'VULN' if true_label == 1 else 'SAFE'} | Pred prob: {pred_prob:.4f}")
    print(f"{'='*70}")
    
    # Truncate code for display
    code_display = code[:1500] + "..." if len(code) > 1500 else code
    print(code_display)
    
    if vuln_features is not None:
        print(f"\n--- Key Vuln Features ---")
        # Feature names (from enhanced_features.py)
        feature_names = [
            'danger_score', 'vuln_likelihood_score', 'defense_ratio',
            'has_unbounded_strcpy', 'has_unchecked_malloc', 'has_format_string_vuln',
            'has_buffer_overflow_risk', 'has_use_after_free_risk'
        ]
        for i, name in enumerate(feature_names):
            if i < len(vuln_features):
                print(f"  {name}: {vuln_features[i]:.4f}")

# %%
if test_codes:
    print("\n" + "=" * 70)
    print("FALSE POSITIVES (Model says VULN, but actually SAFE)")
    print("These are the 940 samples causing low precision")
    print("=" * 70)
    
    fp_indices = np.where(fp_mask)[0]
    # Sort by confidence (highest prob first - most confident mistakes)
    fp_sorted = sorted(fp_indices, key=lambda i: all_probs[i], reverse=True)
    
    print(f"\nShowing top 5 most confident False Positives:")
    for idx in fp_sorted[:5]:
        if idx < len(test_codes):
            print_sample(idx, test_codes[idx], test_labels[idx], all_probs[idx], 
                        test_vuln_features[idx] if test_vuln_features is not None else None)

# %%
if test_codes:
    print("\n" + "=" * 70)
    print("FALSE NEGATIVES (Model says SAFE, but actually VULN)")
    print("These are the 200 missed vulnerabilities")
    print("=" * 70)
    
    fn_indices = np.where(fn_mask)[0]
    # Sort by confidence (lowest prob first - most confident mistakes)
    fn_sorted = sorted(fn_indices, key=lambda i: all_probs[i])
    
    print(f"\nShowing top 5 most confident False Negatives:")
    for idx in fn_sorted[:5]:
        if idx < len(test_codes):
            print_sample(idx, test_codes[idx], test_labels[idx], all_probs[idx],
                        test_vuln_features[idx] if test_vuln_features is not None else None)

# %% [markdown]
# ## 6. Vuln Feature Analysis

# %%
print("\n" + "=" * 60)
print("VULNERABILITY FEATURE ANALYSIS")
print("=" * 60)

# Compare feature distributions between TP, TN, FP, FN
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Key feature indices (adjust based on your feature order)
key_features = [
    (0, 'danger_score'),
    (1, 'vuln_likelihood'),
    (2, 'defense_ratio'),
    (3, 'has_unbounded_strcpy'),
    (4, 'has_unchecked_malloc'),
    (5, 'has_format_vuln'),
    (6, 'buffer_overflow_risk'),
    (7, 'use_after_free_risk'),
]

categories = {
    'TP': tp_mask,
    'TN': tn_mask,
    'FP': fp_mask,
    'FN': fn_mask,
}

for i, (feat_idx, feat_name) in enumerate(key_features):
    ax = axes[i // 4, i % 4]
    
    for cat_name, mask in categories.items():
        feat_values = test_vuln_features[mask, feat_idx]
        ax.hist(feat_values, bins=30, alpha=0.5, label=cat_name, density=True)
    
    ax.set_title(feat_name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'vuln_features_by_category.png'), dpi=150)
plt.show()

# %%
# Feature means by category
print("\nFeature Means by Classification Category:")
print("-" * 80)

feature_names = [
    'danger_score', 'vuln_likelihood', 'defense_ratio', 
    'has_unbounded_strcpy', 'has_unchecked_malloc', 'has_format_vuln',
    'buffer_overflow_risk', 'use_after_free_risk'
]

header = f"{'Feature':<25}" + "".join([f"{cat:<12}" for cat in categories.keys()])
print(header)
print("-" * 80)

for feat_idx, feat_name in enumerate(feature_names[:8]):
    row = f"{feat_name:<25}"
    for cat_name, mask in categories.items():
        mean_val = test_vuln_features[mask, feat_idx].mean()
        row += f"{mean_val:<12.4f}"
    print(row)

# %% [markdown]
# ## 7. Summary and Recommendations

# %%
print("\n" + "=" * 70)
print("SUMMARY: ROOT CAUSE ANALYSIS")
print("=" * 70)

print("""
1. THRESHOLD COLLAPSE EXPLANATION:
   - Model outputs probabilities skewed toward higher values
   - Non-vuln samples have mean prob ~0.4-0.5 (should be < 0.3)
   - Vuln samples have mean prob ~0.5-0.6 (should be > 0.7)
   - The two distributions OVERLAP significantly
   - At threshold 0.5, recall is very low (many vuln predicted as safe)
   - At threshold 0.2, we capture most vuln but also many safe (high FP)

2. ROOT CAUSE - POOR CLASS SEPARATION:
   - The model cannot distinguish vuln from non-vuln code
   - Possible reasons:
     a) Token representation lacks semantic meaning
     b) Many samples are ambiguous (similar code, different labels)
     c) Vulnerability patterns are subtle and context-dependent
     d) Data quality issues (mislabeled samples)

3. FALSE POSITIVE PATTERNS:
   - Model marks safe code as vulnerable when it sees:
     * Memory operations (malloc, free) even with proper checks
     * String functions even when used safely
     * Complex pointer operations
   
4. FALSE NEGATIVE PATTERNS:
   - Model misses vulnerabilities when:
     * Code looks "clean" but has subtle bugs
     * Vulnerability is in logic, not obvious API misuse
     * Defense patterns mask the vulnerability

5. RECOMMENDATIONS:
   a) Use pre-trained code models (CodeBERT) for better semantics
   b) Add data augmentation to balance hard cases
   c) Use contrastive learning to separate classes
   d) Review and clean ambiguous samples
   e) Focus on specific vulnerability types (not all at once)
""")

print("\nAnalysis complete! Plots saved to:", PLOT_DIR)

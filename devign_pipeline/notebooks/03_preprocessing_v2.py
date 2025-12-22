# %% [markdown]
# # Devign Dataset Preprocessing V2
# 
# Cải tiến so với V1:
# 1. **Multi-slice**: backward + forward slicing kết hợp
# 2. **Slice Features**: 22 features tính trên slice + 3 global features (Option A)
# 3. **Hybrid Tokenization**: giữ dangerous APIs, normalize identifiers còn lại
# 
# **Environment**: Kaggle with 2x NVIDIA T4 GPU (32GB total VRAM), 13GB RAM

# %% [markdown]
# ## 1. Setup & Installation
# 
# Run these commands in a separate cell BEFORE running this script:
# ```
# !pip install -q tree-sitter tree-sitter-c tree-sitter-cpp networkx tqdm joblib pyyaml
# ```

# %%
# Setup paths (no copy needed - import directly from input)
import subprocess
import sys

# Add pipeline to path
sys.path.insert(0, '/kaggle/input/devign-pipeline/devign_pipeline')

# Check GPU
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("PyTorch not available")

# %% [markdown]
# ## 2. Configuration

# %%
import sys
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Kaggle paths
DATA_DIR = '/kaggle/input/devign'
WORKING_DIR = '/kaggle/working'
# Path already added in Setup section

OUTPUT_DIR = os.path.join(WORKING_DIR, 'processed_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Multi-slice configuration
MULTI_SLICE_CONFIG = {
    'backward_depth': 5,
    'backward_window': 15,
    'forward_depth': 3,
    'forward_window': 10,
    'max_combined_tokens': 512,
    'sep_token': '[SEP]',
    'include_control_deps': True,
    'include_data_deps': True,
}

# Tokenization configuration
TOKENIZER_CONFIG = {
    'preserve_dangerous_apis': True,
    'preserve_keywords': True,
    'max_seq_length': 512,
    'min_freq': 2,
    'max_vocab_size': 500,
}

# Processing configuration
PROCESS_CONFIG = {
    'batch_size': 500,
    'n_jobs': 4,
}

print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 3. Load Data

# %%
from src.data.loader import DevignLoader

loader = DevignLoader(DATA_DIR, chunk_size=2000)
splits = loader.get_splits()
print("Dataset splits:", splits)

# Load all splits
train_df = loader.load_all(split='train')
val_df = loader.load_all(split='validation')
test_df = loader.load_all(split='test')

print(f"\nTrain set: {len(train_df)} samples")
print(f"Val set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Label distribution
print(f"\nTrain label distribution:")
print(f"  Non-vulnerable: {(train_df['target'] == 0).sum()}")
print(f"  Vulnerable: {(train_df['target'] == 1).sum()}")

# %% [markdown]
# ## 4. Step 1: Multi-slicing (Backward + Forward)

# %%
from src.slicing.multi_slicer import MultiCodeSlicer, MultiSliceConfig, multi_slice_batch
from src.vuln.dictionary import VulnDictionary

# Initialize multi-slicer
multi_slice_config = MultiSliceConfig(**MULTI_SLICE_CONFIG)
slicer = MultiCodeSlicer(multi_slice_config)

# Initialize dictionary for criterion detection
vuln_dict = VulnDictionary()


def find_criterion_lines(code: str, dictionary: VulnDictionary) -> list:
    """Find criterion lines based on dangerous API calls and patterns."""
    import re
    
    lines = code.split('\n')
    criterion_lines = []
    
    dangerous_apis = dictionary.get_all_dangerous_functions()
    
    for i, line in enumerate(lines, 1):
        for api in dangerous_apis:
            if re.search(rf'\b{api}\s*\(', line):
                criterion_lines.append(i)
                break
    
    if not criterion_lines and lines:
        criterion_lines = [len(lines) // 2] if len(lines) > 1 else [1]
    
    return criterion_lines


def process_multi_slice_batch(df, batch_size=500):
    """Process multi-slicing in batches."""
    codes = df['func'].tolist()
    n_samples = len(codes)
    
    slice_results = []
    combined_codes = []
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Multi-slicing"):
        end = min(start + batch_size, n_samples)
        batch_codes = codes[start:end]
        
        batch_criteria = [find_criterion_lines(code, vuln_dict) for code in batch_codes]
        
        for code, criteria in zip(batch_codes, batch_criteria):
            try:
                result = slicer.multi_slice(code, criteria)
                slice_results.append(result)
                combined_codes.append(result.combined_code)
            except Exception as e:
                slice_results.append(None)
                combined_codes.append(code)
    
    return slice_results, combined_codes


# Process all splits
print("Processing train set...")
train_slice_results, train_combined = process_multi_slice_batch(train_df, PROCESS_CONFIG['batch_size'])

print("Processing val set...")
val_slice_results, val_combined = process_multi_slice_batch(val_df, PROCESS_CONFIG['batch_size'])

print("Processing test set...")
test_slice_results, test_combined = process_multi_slice_batch(test_df, PROCESS_CONFIG['batch_size'])

print(f"\nMulti-slice completed:")
print(f"  Train: {len(train_combined)} samples")
print(f"  Val: {len(val_combined)} samples")
print(f"  Test: {len(test_combined)} samples")

# %% [markdown]
# ## 5. Step 2: Extract Slice Features

# %%
from src.vuln.slice_features import extract_slice_features, extract_slice_features_batch, SLICE_FEATURE_NAMES

print(f"Feature count: {len(SLICE_FEATURE_NAMES)}")
print(f"Features: {SLICE_FEATURE_NAMES}")


def process_features_batch(combined_codes, original_codes, batch_size=500):
    """Extract features in batches."""
    n_samples = len(combined_codes)
    all_features = []
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Extracting features"):
        end = min(start + batch_size, n_samples)
        batch_slices = combined_codes[start:end]
        batch_fulls = original_codes[start:end]
        
        batch_features = extract_slice_features_batch(
            batch_slices, batch_fulls, vuln_dict, n_jobs=1
        )
        all_features.extend(batch_features)
    
    return all_features


# Process all splits
print("Extracting train features...")
train_features = process_features_batch(train_combined, train_df['func'].tolist(), PROCESS_CONFIG['batch_size'])

print("Extracting val features...")
val_features = process_features_batch(val_combined, val_df['func'].tolist(), PROCESS_CONFIG['batch_size'])

print("Extracting test features...")
test_features = process_features_batch(test_combined, test_df['func'].tolist(), PROCESS_CONFIG['batch_size'])

print(f"\nFeature extraction completed:")
print(f"  Train: {len(train_features)} samples x {len(SLICE_FEATURE_NAMES)} features")
print(f"  Val: {len(val_features)} samples")
print(f"  Test: {len(test_features)} samples")

# Sample feature values
if train_features:
    sample_feat = train_features[0]
    print("\nSample feature values:")
    for name in SLICE_FEATURE_NAMES[:5]:
        print(f"  {name}: {sample_feat[name]:.3f}")

# %% [markdown]
# ## 6. Step 3: Hybrid Tokenization

# %%
from src.tokenization.hybrid_tokenizer import (
    HybridTokenizer, 
    build_hybrid_vocab, 
    vectorize,
    DANGEROUS_APIS
)

# Initialize tokenizer
tokenizer = HybridTokenizer(
    preserve_dangerous_apis=TOKENIZER_CONFIG['preserve_dangerous_apis'],
    preserve_keywords=TOKENIZER_CONFIG['preserve_keywords']
)


def tokenize_batch(codes, batch_size=500):
    """Tokenize codes in batches."""
    n_samples = len(codes)
    all_tokens = []
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Tokenizing"):
        end = min(start + batch_size, n_samples)
        batch_codes = codes[start:end]
        
        batch_tokens = tokenizer.tokenize_batch(batch_codes, n_jobs=1)
        all_tokens.extend(batch_tokens)
    
    return all_tokens


# Tokenize all splits
print("Tokenizing train set...")
train_tokens = tokenize_batch(train_combined, PROCESS_CONFIG['batch_size'])

print("Tokenizing val set...")
val_tokens = tokenize_batch(val_combined, PROCESS_CONFIG['batch_size'])

print("Tokenizing test set...")
test_tokens = tokenize_batch(test_combined, PROCESS_CONFIG['batch_size'])

print(f"\nTokenization completed:")
print(f"  Train: {len(train_tokens)} samples")
print(f"  Val: {len(val_tokens)} samples")
print(f"  Test: {len(test_tokens)} samples")

# Token statistics
train_lens = [len(t) for t in train_tokens]
print(f"\nTrain token length stats:")
print(f"  Mean: {np.mean(train_lens):.1f}")
print(f"  Max: {max(train_lens)}")
print(f"  Min: {min(train_lens)}")

# Sample tokenization
if train_tokens:
    sample = train_tokens[0][:20]
    preserved = [t for t in sample if t in DANGEROUS_APIS]
    print(f"\nSample tokens (first 20): {sample}")
    print(f"Preserved APIs: {preserved}")

# %% [markdown]
# ## 7. Step 4: Build Vocabulary & Vectorize

# %%
# Build vocabulary from train set only
print("Building vocabulary from train set...")
vocab = build_hybrid_vocab(
    train_combined,
    min_freq=TOKENIZER_CONFIG['min_freq'],
    max_size=TOKENIZER_CONFIG['max_vocab_size']
)

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocab entries: {dict(list(vocab.items())[:15])}")

# Check dangerous APIs in vocab
preserved_apis_in_vocab = [api for api in DANGEROUS_APIS if api in vocab]
print(f"\nDangerous APIs in vocab: {len(preserved_apis_in_vocab)}/{len(DANGEROUS_APIS)}")
print(f"APIs: {preserved_apis_in_vocab[:10]}...")


def vectorize_batch(tokens_list, vocab, max_len, batch_size=500):
    """Vectorize tokens in batches."""
    n_samples = len(tokens_list)
    all_input_ids = []
    all_attention_masks = []
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Vectorizing"):
        end = min(start + batch_size, n_samples)
        batch_tokens = tokens_list[start:end]
        
        for tokens in batch_tokens:
            input_ids, attention_mask = vectorize(tokens, vocab, max_len)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
    
    return np.array(all_input_ids, dtype=np.int32), np.array(all_attention_masks, dtype=np.int32)


# Vectorize all splits
max_len = TOKENIZER_CONFIG['max_seq_length']

print("\nVectorizing train set...")
train_input_ids, train_attention_mask = vectorize_batch(train_tokens, vocab, max_len, PROCESS_CONFIG['batch_size'])

print("Vectorizing val set...")
val_input_ids, val_attention_mask = vectorize_batch(val_tokens, vocab, max_len, PROCESS_CONFIG['batch_size'])

print("Vectorizing test set...")
test_input_ids, test_attention_mask = vectorize_batch(test_tokens, vocab, max_len, PROCESS_CONFIG['batch_size'])

print(f"\nVectorization completed:")
print(f"  Train: input_ids {train_input_ids.shape}, attention_mask {train_attention_mask.shape}")
print(f"  Val: input_ids {val_input_ids.shape}")
print(f"  Test: input_ids {test_input_ids.shape}")

# %% [markdown]
# ## 8. Step 5: Save Output

# %%
def features_to_array(features_list):
    """Convert list of feature dicts to numpy array."""
    n_samples = len(features_list)
    n_features = len(SLICE_FEATURE_NAMES)
    
    arr = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for i, feat_dict in enumerate(features_list):
        for j, name in enumerate(SLICE_FEATURE_NAMES):
            arr[i, j] = feat_dict.get(name, 0.0)
    
    return arr


# Convert features to arrays
print("Converting features to arrays...")
train_vuln_features = features_to_array(train_features)
val_vuln_features = features_to_array(val_features)
test_vuln_features = features_to_array(test_features)

print(f"  Train: {train_vuln_features.shape}")
print(f"  Val: {val_vuln_features.shape}")
print(f"  Test: {test_vuln_features.shape}")

# Get labels
train_labels = train_df['target'].values.astype(np.int32)
val_labels = val_df['target'].values.astype(np.int32)
test_labels = test_df['target'].values.astype(np.int32)

# Save sequence data (input_ids, attention_mask, labels)
print("\nSaving sequence data...")
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'train.npz'),
    input_ids=train_input_ids,
    attention_mask=train_attention_mask,
    labels=train_labels
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'val.npz'),
    input_ids=val_input_ids,
    attention_mask=val_attention_mask,
    labels=val_labels
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'test.npz'),
    input_ids=test_input_ids,
    attention_mask=test_attention_mask,
    labels=test_labels
)

# Save vulnerability features
print("Saving vulnerability features...")
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'train_vuln.npz'),
    features=train_vuln_features,
    feature_names=SLICE_FEATURE_NAMES
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'val_vuln.npz'),
    features=val_vuln_features,
    feature_names=SLICE_FEATURE_NAMES
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'test_vuln.npz'),
    features=test_vuln_features,
    feature_names=SLICE_FEATURE_NAMES
)

# Save vocabulary
print("Saving vocabulary...")
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'w') as f:
    json.dump(vocab, f, indent=2)

# Save configuration
print("Saving configuration...")
config = {
    'multi_slice': MULTI_SLICE_CONFIG,
    'tokenizer': TOKENIZER_CONFIG,
    'process': PROCESS_CONFIG,
    'vocab_size': len(vocab),
    'n_features': len(SLICE_FEATURE_NAMES),
    'feature_names': SLICE_FEATURE_NAMES,
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
}

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\n✓ All outputs saved!")

# %% [markdown]
# ## 9. Verify Output

# %%
print("=" * 60)
print("OUTPUT VERIFICATION")
print("=" * 60)

# List output files
output_files = list(Path(OUTPUT_DIR).glob('*'))
print(f"\nOutput files ({len(output_files)}):")
for f in sorted(output_files):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {size_mb:.2f} MB")

# Verify sequence data
print("\n--- Sequence Data ---")
for split in ['train', 'val', 'test']:
    data = np.load(os.path.join(OUTPUT_DIR, f'{split}.npz'))
    print(f"\n{split}.npz:")
    print(f"  input_ids: {data['input_ids'].shape}")
    print(f"  attention_mask: {data['attention_mask'].shape}")
    print(f"  labels: {data['labels'].shape}")
    print(f"  Label distribution: 0={np.sum(data['labels']==0)}, 1={np.sum(data['labels']==1)}")

# Verify vulnerability features
print("\n--- Vulnerability Features ---")
for split in ['train', 'val', 'test']:
    data = np.load(os.path.join(OUTPUT_DIR, f'{split}_vuln.npz'), allow_pickle=True)
    features = data['features']
    print(f"\n{split}_vuln.npz:")
    print(f"  features: {features.shape}")
    print(f"  Non-zero features: {np.count_nonzero(features)}/{features.size} ({100*np.count_nonzero(features)/features.size:.1f}%)")
    
    # Feature statistics
    if features.size > 0:
        print(f"  Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")

# Verify vocab
print("\n--- Vocabulary ---")
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'r') as f:
    loaded_vocab = json.load(f)
print(f"  Size: {len(loaded_vocab)}")
print(f"  Sample: {dict(list(loaded_vocab.items())[:10])}")

# Verify config
print("\n--- Configuration ---")
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'r') as f:
    loaded_config = json.load(f)
print(f"  vocab_size: {loaded_config['vocab_size']}")
print(f"  n_features: {loaded_config['n_features']}")
print(f"  Splits: train={loaded_config['train_samples']}, val={loaded_config['val_samples']}, test={loaded_config['test_samples']}")

print("\n" + "=" * 60)
print("✓ VERIFICATION COMPLETE!")
print("=" * 60)

# %% [markdown]
# ## Next Steps
# 
# 1. Use `04_training_v2.ipynb` to train with combined sequence + features
# 2. Model architecture:
#    - BiGRU for sequence encoding
#    - MLP for vulnerability features
#    - Fusion layer to combine both
# 3. Or use features only with XGBoost/LightGBM baseline

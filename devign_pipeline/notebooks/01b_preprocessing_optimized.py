# %% [markdown]
# # Devign Dataset Preprocessing with Optimized Tokenizer
# 
# This notebook preprocesses Devign dataset using OptimizedHybridTokenizer.
# 
# **Key differences from original preprocessing:**
# - API family mapping (130+ APIs -> ~20 families)
# - Semantic buckets for identifiers (BUF_0, LEN_1, PTR_0...)
# - Defense pattern detection
# - Much smaller vocab (~2k vs 30k)
#
# **Run on Kaggle after uploading devign_pipeline as dataset**

# %%
# Dependencies: Ensure these are installed via requirements.txt or Dockerfile
# Required: tree-sitter, tree-sitter-c, tree-sitter-cpp, networkx, tqdm, joblib, pyyaml
# On Kaggle, these are pre-installed or use: !pip install -q <package>

# %%
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Detect environment
if os.path.exists('/kaggle'):
    KAGGLE = True
    # Copy pipeline to working dir
    if not os.path.exists('/kaggle/working/devign_pipeline'):
        os.system('cp -r /kaggle/input/devign-pipeline/devign_pipeline /kaggle/working/')
    
    DATA_DIR = '/kaggle/input/devign'
    PROCESSED_INPUT = '/kaggle/input/devign-processed/processed'  # Previous processed data
    WORKING_DIR = '/kaggle/working'
    sys.path.insert(0, '/kaggle/working/devign_pipeline')
else:
    KAGGLE = False
    DATA_DIR = 'f:/Work/C Vul Devign/Dataset'
    PROCESSED_INPUT = 'f:/Work/C Vul Devign/output/processed'
    WORKING_DIR = 'f:/Work/C Vul Devign/output'
    sys.path.insert(0, 'f:/Work/C Vul Devign/devign_pipeline')

OUTPUT_DIR = os.path.join(WORKING_DIR, 'processed_optimized')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Running on {'Kaggle' if KAGGLE else 'Local'}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# %%
from src.tokenization.optimized_tokenizer import (
    OptimizedHybridTokenizer,
    build_optimized_vocab,
    vectorize_optimized,
    API_FAMILIES,
    DEFENSE_FAMILIES,
    SEMANTIC_BUCKETS,
)
from src.slicing.slicer import VulnerabilityAwareSlicer

print("Imports successful!")
print(f"API Families: {len(API_FAMILIES)}")
print(f"Semantic Buckets: {len(SEMANTIC_BUCKETS)}")

# %% [markdown]
# ## Load Previous Sliced Data
# 
# We'll use the already-sliced code from previous preprocessing, 
# just re-tokenize with OptimizedHybridTokenizer.

# %%
# Try to load previously processed slices
def load_sliced_data(processed_dir, split='train'):
    """Load sliced code from previous preprocessing."""
    slice_file = os.path.join(processed_dir, f'{split}_slices.json')
    
    if os.path.exists(slice_file):
        with open(slice_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} slices from {slice_file}")
        return data
    
    # Alternative: load from raw Devign and slice
    logger.info(f"No slices found at {slice_file}, will need to slice from raw data")
    return None

# Check what's available
for split in ['train', 'val', 'test']:
    slice_file = os.path.join(PROCESSED_INPUT, f'{split}_slices.json')
    if os.path.exists(slice_file):
        print(f"✓ Found {split}_slices.json")
    else:
        print(f"✗ Missing {split}_slices.json")

# %% [markdown]
# ## Option A: Re-tokenize from existing slices

# %%
def retokenize_from_slices(processed_dir, output_dir, split='train'):
    """
    Re-tokenize previously sliced code using OptimizedHybridTokenizer.
    """
    slice_file = os.path.join(processed_dir, f'{split}_slices.json')
    
    if not os.path.exists(slice_file):
        logger.error(f"Slice file not found: {slice_file}")
        return None
    
    with open(slice_file, 'r') as f:
        slices_data = json.load(f)
    
    logger.info(f"Loaded {len(slices_data)} samples for {split}")
    
    tokenizer = OptimizedHybridTokenizer()
    
    all_tokens = []
    labels = []
    
    for item in tqdm(slices_data, desc=f"Tokenizing {split}"):
        code = item.get('sliced_code', item.get('code', ''))
        label = item.get('label', item.get('target', 0))
        
        tokens = tokenizer.tokenize(code)
        all_tokens.append(tokens)
        labels.append(label)
    
    logger.info(f"Tokenized {len(all_tokens)} samples")
    
    return all_tokens, labels

# %% [markdown]
# ## Option B: Full pipeline from raw Devign data

# %%
def process_from_raw(data_dir, output_dir, split='train', max_samples=None):
    """
    Full pipeline: load raw -> slice -> tokenize with OptimizedHybridTokenizer
    """
    import pandas as pd
    
    # Load raw data
    json_file = os.path.join(data_dir, f'{split}.json') 
    if not os.path.exists(json_file):
        # Try alternative naming
        json_file = os.path.join(data_dir, 'function.json')
    
    if not os.path.exists(json_file):
        logger.error(f"Data file not found: {json_file}")
        return None, None
    
    df = pd.read_json(json_file)
    logger.info(f"Loaded {len(df)} samples from {json_file}")
    
    if max_samples:
        df = df.head(max_samples)
    
    # Initialize components
    slicer = VulnerabilityAwareSlicer()
    tokenizer = OptimizedHybridTokenizer()
    
    all_tokens = []
    labels = []
    slices_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        code = row.get('func', row.get('code', ''))
        label = row.get('target', row.get('label', 0))
        
        try:
            # Slice
            sliced_code = slicer.slice_function(code)
            
            # Tokenize
            tokens = tokenizer.tokenize(sliced_code)
            
            all_tokens.append(tokens)
            labels.append(label)
            slices_data.append({
                'idx': idx,
                'sliced_code': sliced_code,
                'label': label
            })
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # Use original code as fallback
            tokens = tokenizer.tokenize(code)
            all_tokens.append(tokens)
            labels.append(label)
    
    # Save slices for future use
    slice_file = os.path.join(output_dir, f'{split}_slices.json')
    with open(slice_file, 'w') as f:
        json.dump(slices_data, f)
    logger.info(f"Saved slices to {slice_file}")
    
    return all_tokens, labels

# %% [markdown]
# ## Run Tokenization

# %%
# Choose method based on available data
train_tokens, train_labels = None, None
val_tokens, val_labels = None, None  
test_tokens, test_labels = None, None

# Try Option A first (re-tokenize from slices)
slice_file = os.path.join(PROCESSED_INPUT, 'train_slices.json')

if os.path.exists(slice_file):
    logger.info("Using Option A: Re-tokenizing from existing slices")
    train_tokens, train_labels = retokenize_from_slices(PROCESSED_INPUT, OUTPUT_DIR, 'train')
    val_tokens, val_labels = retokenize_from_slices(PROCESSED_INPUT, OUTPUT_DIR, 'val')
    test_tokens, test_labels = retokenize_from_slices(PROCESSED_INPUT, OUTPUT_DIR, 'test')
else:
    logger.info("Using Option B: Full pipeline from raw data")
    train_tokens, train_labels = process_from_raw(DATA_DIR, OUTPUT_DIR, 'train')
    val_tokens, val_labels = process_from_raw(DATA_DIR, OUTPUT_DIR, 'val')
    test_tokens, test_labels = process_from_raw(DATA_DIR, OUTPUT_DIR, 'test')

# %%
# Token statistics
if train_tokens:
    print("\n=== Token Statistics ===")
    all_tokens_flat = [t for tokens in train_tokens for t in tokens]
    token_counts = Counter(all_tokens_flat)
    
    print(f"Total tokens: {len(all_tokens_flat)}")
    print(f"Unique tokens: {len(token_counts)}")
    print(f"Avg sequence length: {np.mean([len(t) for t in train_tokens]):.1f}")
    
    print("\nTop 30 tokens:")
    for tok, count in token_counts.most_common(30):
        print(f"  {tok}: {count}")
    
    # API family distribution
    print("\nAPI Family tokens:")
    api_tokens = {t: c for t, c in token_counts.items() if t.startswith('API_')}
    for tok, count in sorted(api_tokens.items(), key=lambda x: -x[1]):
        print(f"  {tok}: {count}")
    
    # Defense tokens
    print("\nDefense tokens:")
    def_tokens = {t: c for t, c in token_counts.items() if t.startswith('DEF_')}
    for tok, count in sorted(def_tokens.items(), key=lambda x: -x[1]):
        print(f"  {tok}: {count}")
    
    # Semantic bucket distribution
    print("\nSemantic bucket tokens (sample):")
    bucket_prefixes = ['BUF_', 'LEN_', 'PTR_', 'IDX_', 'CAP_', 'RET_', 'ERR_', 'VAR_']
    for prefix in bucket_prefixes:
        bucket_tokens = {t: c for t, c in token_counts.items() if t.startswith(prefix)}
        total = sum(bucket_tokens.values())
        unique = len(bucket_tokens)
        print(f"  {prefix}: {total} occurrences, {unique} unique")

# %% [markdown]
# ## Build Vocabulary

# %%
if train_tokens:
    logger.info("Building optimized vocabulary...")
    
    vocab, vocab_debug = build_optimized_vocab(
        tokens_list=train_tokens,
        min_freq=2,
        max_size=3000,  # Much smaller than 30k!
    )
    
    print(f"\n=== Vocabulary Statistics ===")
    for key, val in vocab_debug.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    # Save vocabulary
    vocab_file = os.path.join(OUTPUT_DIR, 'vocab.json')
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"Saved vocabulary to {vocab_file}")

# %% [markdown]
# ## Vectorize Data

# %%
def vectorize_split(tokens_list, labels, vocab, split_name, output_dir, max_len=512):
    """Vectorize a split and save to npz."""
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    unk_count = 0
    total_count = 0
    
    for tokens, label in tqdm(zip(tokens_list, labels), total=len(tokens_list), desc=f"Vectorizing {split_name}"):
        input_ids, attention_mask, unk_positions = vectorize_optimized(
            tokens, vocab, max_len=max_len,
            truncation_strategy='head_tail',
            head_tokens=192,
            tail_tokens=319
        )
        
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(label)
        
        unk_count += len(unk_positions)
        total_count += sum(attention_mask)
    
    # Convert to numpy
    input_ids_arr = np.array(all_input_ids, dtype=np.int32)
    attention_masks_arr = np.array(all_attention_masks, dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int32)
    
    # Save
    output_file = os.path.join(output_dir, f'{split_name}.npz')
    np.savez_compressed(
        output_file,
        input_ids=input_ids_arr,
        attention_mask=attention_masks_arr,
        labels=labels_arr
    )
    
    unk_rate = unk_count / total_count if total_count > 0 else 0
    
    logger.info(f"Saved {split_name}: {len(labels)} samples, UNK rate: {unk_rate:.4f}")
    
    return {
        'samples': len(labels),
        'unk_rate': unk_rate,
        'shape': input_ids_arr.shape
    }

# %%
if train_tokens and vocab:
    stats = {}
    
    stats['train'] = vectorize_split(train_tokens, train_labels, vocab, 'train', OUTPUT_DIR)
    
    if val_tokens:
        stats['val'] = vectorize_split(val_tokens, val_labels, vocab, 'val', OUTPUT_DIR)
    
    if test_tokens:
        stats['test'] = vectorize_split(test_tokens, test_labels, vocab, 'test', OUTPUT_DIR)
    
    print("\n=== Final Statistics ===")
    for split, s in stats.items():
        print(f"{split}: {s['samples']} samples, shape {s['shape']}, UNK rate {s['unk_rate']:.4f}")

# %% [markdown]
# ## Save Config

# %%
config = {
    'tokenizer': 'OptimizedHybridTokenizer',
    'vocab_size': len(vocab) if vocab else 0,
    'max_seq_length': 512,
    'truncation_strategy': 'head_tail',
    'head_tokens': 192,
    'tail_tokens': 319,
    'embedding_dim': 128,
    'api_families': len(API_FAMILIES),
    'semantic_buckets': len(SEMANTIC_BUCKETS),
    'preprocessing_date': str(np.datetime64('now')),
}

config_file = os.path.join(OUTPUT_DIR, 'config.json')
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Saved config to {config_file}")
print(json.dumps(config, indent=2))

# %% [markdown]
# ## Verify Output

# %%
# Verify files
print("\n=== Output Files ===")
for f in Path(OUTPUT_DIR).glob('*'):
    size = f.stat().st_size / 1024
    print(f"  {f.name}: {size:.1f} KB")

# Load and verify
train_data = np.load(os.path.join(OUTPUT_DIR, 'train.npz'))
print(f"\nTrain data keys: {list(train_data.keys())}")
print(f"input_ids shape: {train_data['input_ids'].shape}")
print(f"labels shape: {train_data['labels'].shape}")
print(f"Label distribution: 0={np.sum(train_data['labels']==0)}, 1={np.sum(train_data['labels']==1)}")

# %% [markdown]
# ## Next Steps
# 
# 1. Train BiGRU model (use 02_training.py)
# 2. Expected improvements:
#    - Lower UNK rate
#    - Better precision (fewer FP)
#    - More stable threshold

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nNext: Run training with 02_training.py")

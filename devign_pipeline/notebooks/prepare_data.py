# %% [markdown]
# # Full Pipeline: Devign Vulnerability Detection
# 
# Complete pipeline from raw data to training-ready vectors:
# 1. **Load**: Raw Devign dataset from parquet files
# 2. **Slice**: Multi-slice (backward + forward) based on dangerous APIs
# 3. **Tokenize**: PreserveIdentifierTokenizer (keeps variable names, smart number handling)
# 4. **Vectorize**: Build vocab, convert to input_ids with attention masks
# 5. **Save**: NPZ files for training, JSON for debugging
#
# **Environment**: Kaggle with 2x NVIDIA T4 GPU or Local

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import sys
import os
import re
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add pipeline to path - auto-detect environment
NOTEBOOK_DIR = Path(__file__).parent if '__file__' in dir() else Path.cwd()
PIPELINE_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR

if os.path.exists('/kaggle/input'):
    # Kaggle environment
    sys.path.insert(0, '/kaggle/input/devign-pipeline/devign_pipeline')
    DATA_DIR = '/kaggle/input/devign'
    WORKING_DIR = '/kaggle/working'
else:
    # Local environment
    sys.path.insert(0, str(PIPELINE_ROOT))
    DATA_DIR = os.environ.get('DEVIGN_DATA_DIR', str(PIPELINE_ROOT.parent / 'Dataset' / 'devign'))
    WORKING_DIR = os.environ.get('DEVIGN_OUTPUT_DIR', str(PIPELINE_ROOT.parent / 'output'))

OUTPUT_DIR = os.path.join(WORKING_DIR, 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Pipeline root: {PIPELINE_ROOT}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 2. Configuration

# %%
# Multi-slice configuration
SLICE_CONFIG = {
    'backward_depth': 5,
    'backward_window': 15,
    'forward_depth': 3,
    'forward_window': 10,
    'max_combined_tokens': 512,
    'sep_token': '[SEP]',
    'include_control_deps': True,
    'include_data_deps': True,
    'remove_comments': True,
    'normalize_output': True,
}

# Tokenization configuration
TOKENIZER_CONFIG = {
    'min_freq': 3,
    'max_vocab_size': 10000,
    'max_seq_length': 512,
    'preserve_identifiers': True,
    'preserve_dangerous_apis': True,
    'preserve_keywords': True,
    'numeric_policy': {
        'keep_small_integers': True,
        'keep_negative_one': True,
        'keep_power_of_two': True,
        'keep_common_sizes': True,
        'keep_hex_masks': True,
        'keep_permissions': True,
    }
}

# Processing configuration
PROCESS_CONFIG = {
    'batch_size': 500,
    'min_slice_tokens': 10,  # Fallback to full code if slice too short
    'n_jobs': 1,
}

print("=== Configuration ===")
print(f"Slice config: {json.dumps(SLICE_CONFIG, indent=2)}")
print(f"Tokenizer config: {json.dumps(TOKENIZER_CONFIG, indent=2)}")

# %% [markdown]
# ## 3. Import Modules

# %%
from src.data.loader import DevignLoader
from src.slicing.multi_slicer import MultiCodeSlicer, MultiSliceConfig
from src.tokenization.preserve_tokenizer import (
    PreserveIdentifierTokenizer,
    build_preserve_vocab,
    vectorize_batch_preserve,
    FORCE_KEEP_IDENTIFIERS,
    PRESERVED_NUMBERS,
    PRESERVED_HEX,
    PRESERVED_OCTAL,
)
from src.tokenization.hybrid_tokenizer import DANGEROUS_APIS, SPECIAL_TOKENS
from src.slicing.utils import find_criterion_lines

print("All modules imported successfully!")

# %% [markdown]
# ## 4. Load Raw Data

# %%
print("\n" + "=" * 60)
print("STEP 1: LOADING RAW DATA")
print("=" * 60)

loader = DevignLoader(DATA_DIR, chunk_size=2000)
train_df = loader.load_all(split='train')
val_df = loader.load_all(split='validation')
test_df = loader.load_all(split='test')

# Validate columns exist and handle NaN
for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    assert 'func' in df.columns, f"Missing 'func' column in {split_name}"
    assert 'target' in df.columns, f"Missing 'target' column in {split_name}"
    nan_count = df['func'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {split_name} has {nan_count} NaN values in 'func' column, replacing with empty string")
        df['func'] = df['func'].fillna('')

print(f"\nDataset loaded:")
print(f"  Train: {len(train_df)} samples (vuln: {(train_df['target']==1).sum()})")
print(f"  Val: {len(val_df)} samples (vuln: {(val_df['target']==1).sum()})")
print(f"  Test: {len(test_df)} samples (vuln: {(test_df['target']==1).sum()})")

# %% [markdown]
# ## 5. Multi-Slicing

# %%
print("\n" + "=" * 60)
print("STEP 2: MULTI-SLICING (BACKWARD + FORWARD)")
print("=" * 60)

# Initialize multi-slicer
multi_slice_config = MultiSliceConfig(**SLICE_CONFIG)
slicer = MultiCodeSlicer(multi_slice_config)


def estimate_tokens(code: str) -> int:
    """Rough token count estimate."""
    return len(re.findall(r'\b\w+\b|[^\s\w]', code))


def insert_sep_in_middle(code: str, sep_token: str = "[SEP]") -> str:
    """Insert SEP token in the middle of code for fallback cases."""
    lines = code.strip().split('\n')
    if len(lines) <= 1:
        return f"{code.strip()} {sep_token}"
    mid = len(lines) // 2
    before = '\n'.join(lines[:mid])
    after = '\n'.join(lines[mid:])
    return f"{before} {sep_token} {after}"


def process_multi_slice_batch(df, batch_size=500):
    """Process multi-slicing in batches with fallback for short slices."""
    codes = df['func'].tolist()
    n_samples = len(codes)
    
    combined_codes = []
    fallback_count = 0
    slice_stats = {'backward': 0, 'forward': 0, 'window': 0}
    
    for start in tqdm(range(0, n_samples, batch_size), desc="Multi-slicing"):
        end = min(start + batch_size, n_samples)
        batch_codes = codes[start:end]
        batch_criteria = [find_criterion_lines(code) for code in batch_codes]
        
        for code, criteria in zip(batch_codes, batch_criteria):
            try:
                result = slicer.multi_slice(code, criteria)
                combined = result.combined_code
                
                # Track slice types
                if result.backward_slice.slice_type.value == 'window':
                    slice_stats['window'] += 1
                else:
                    slice_stats['backward'] += 1
                    slice_stats['forward'] += 1
                
                # Check if slice is too short - fallback to full function with SEP
                if estimate_tokens(combined) < PROCESS_CONFIG['min_slice_tokens']:
                    combined = insert_sep_in_middle(code)
                    fallback_count += 1
                
                combined_codes.append(combined)
            except Exception:
                combined_codes.append(insert_sep_in_middle(code))
                fallback_count += 1
    
    print(f"  Graph-based slices: {slice_stats['backward']}")
    print(f"  Window fallback: {slice_stats['window']}")
    print(f"  Too-short fallback: {fallback_count}")
    
    return combined_codes


# Process all splits
print("\nProcessing train set...")
train_sliced = process_multi_slice_batch(train_df, PROCESS_CONFIG['batch_size'])

print("\nProcessing val set...")
val_sliced = process_multi_slice_batch(val_df, PROCESS_CONFIG['batch_size'])

print("\nProcessing test set...")
test_sliced = process_multi_slice_batch(test_df, PROCESS_CONFIG['batch_size'])

print(f"\nMulti-slice completed:")
print(f"  Train: {len(train_sliced)} samples")
print(f"  Val: {len(val_sliced)} samples")
print(f"  Test: {len(test_sliced)} samples")

# %% [markdown]
# ## 6. Tokenization

# %%
print("\n" + "=" * 60)
print("STEP 3: TOKENIZATION")
print("=" * 60)

tokenizer = PreserveIdentifierTokenizer(TOKENIZER_CONFIG)

# Tokenize all splits
print("\nTokenizing train set...")
train_tokens, train_details, train_stats = tokenizer.tokenize_batch(
    train_sliced, with_details=True
)

print("\nTokenizing val set...")
val_tokens, val_details, val_stats = tokenizer.tokenize_batch(
    val_sliced, with_details=True
)

print("\nTokenizing test set...")
test_tokens, test_details, test_stats = tokenizer.tokenize_batch(
    test_sliced, with_details=True
)

# Statistics
train_lens = [len(t) for t in train_tokens]
print(f"\nTokenization completed:")
print(f"  Train: {len(train_tokens)} samples")
print(f"  Val: {len(val_tokens)} samples")
print(f"  Test: {len(test_tokens)} samples")
print(f"\nToken length stats (train):")
print(f"  Mean: {np.mean(train_lens):.1f}")
print(f"  Median: {np.median(train_lens):.1f}")
print(f"  Max: {max(train_lens) if train_lens else 0}")
print(f"  Min: {min(train_lens) if train_lens else 0}")

# %% [markdown]
# ## 7. Build Vocabulary

# %%
print("\n" + "=" * 60)
print("STEP 4: BUILD VOCABULARY")
print("=" * 60)

vocab, vocab_debug = build_preserve_vocab(
    train_tokens,
    min_freq=TOKENIZER_CONFIG['min_freq'],
    max_size=TOKENIZER_CONFIG['max_vocab_size'],
    force_keep=FORCE_KEEP_IDENTIFIERS
)

print(f"\nVocabulary Statistics:")
print(f"  Total unique tokens in train: {vocab_debug['total_unique_tokens']}")
print(f"  Vocab size: {vocab_debug['vocab_size']}")
print(f"  Coverage: {vocab_debug['coverage']:.2%}")
print(f"  Dropped by min_freq ({TOKENIZER_CONFIG['min_freq']}): {vocab_debug['dropped_by_min_freq']}")
print(f"  Dropped by max_size ({TOKENIZER_CONFIG['max_vocab_size']}): {vocab_debug['dropped_by_max_size']}")

print(f"\nTop tokens added (by frequency):")
for tok, count in vocab_debug['top_tokens_added'][:15]:
    print(f"  {tok}: {count}")

# Verify important tokens
print(f"\nVerification:")
important_tokens = ['PAD', 'UNK', 'SEP', 'buf', 'len', 'malloc', 'strcpy', 'NEG_1', '0', '1', '256']
for tok in important_tokens:
    status = f"ID={vocab[tok]}" if tok in vocab else "NOT IN VOCAB"
    print(f"  '{tok}': {status}")

# %% [markdown]
# ## 8. Vectorize Data

# %%
print("\n" + "=" * 60)
print("STEP 5: VECTORIZATION")
print("=" * 60)

max_len = TOKENIZER_CONFIG['max_seq_length']

print("\nVectorizing train set...")
train_input_ids, train_attention_mask, train_unk_pos, train_vec_stats = vectorize_batch_preserve(
    train_tokens, vocab, max_len
)

print("\nVectorizing val set...")
val_input_ids, val_attention_mask, val_unk_pos, val_vec_stats = vectorize_batch_preserve(
    val_tokens, vocab, max_len
)

print("\nVectorizing test set...")
test_input_ids, test_attention_mask, test_unk_pos, test_vec_stats = vectorize_batch_preserve(
    test_tokens, vocab, max_len
)

print(f"\nVectorization Statistics:")
print(f"  Train: {train_input_ids.shape}, UNK rate: {train_vec_stats['unk_rate']:.2%}")
print(f"  Val: {val_input_ids.shape}, UNK rate: {val_vec_stats['unk_rate']:.2%}")
print(f"  Test: {test_input_ids.shape}, UNK rate: {test_vec_stats['unk_rate']:.2%}")

print(f"\nTop UNK tokens (train):")
for tok, count in train_vec_stats['top_unk_tokens'][:15]:
    print(f"  {tok}: {count}")

# %% [markdown]
# ## 9. Save Outputs

# %%
print("\n" + "=" * 60)
print("STEP 6: SAVING OUTPUTS")
print("=" * 60)

# === 1. Save NPZ files ===
train_labels = train_df['target'].values.astype(np.int32)
val_labels = val_df['target'].values.astype(np.int32)
test_labels = test_df['target'].values.astype(np.int32)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'train.npz'),
    input_ids=train_input_ids,
    attention_mask=train_attention_mask,
    labels=train_labels
)
print(f"Saved: train.npz {train_input_ids.shape}")

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'val.npz'),
    input_ids=val_input_ids,
    attention_mask=val_attention_mask,
    labels=val_labels
)
print(f"Saved: val.npz {val_input_ids.shape}")

np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'test.npz'),
    input_ids=test_input_ids,
    attention_mask=test_attention_mask,
    labels=test_labels
)
print(f"Saved: test.npz {test_input_ids.shape}")

# === 2. Save vocab.json ===
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
    json.dump(vocab, f, indent=2, ensure_ascii=False)
print(f"Saved: vocab.json ({len(vocab)} tokens)")

# === 3. Save config.json ===
config = {
    'slice_config': SLICE_CONFIG,
    'tokenizer_config': TOKENIZER_CONFIG,
    'vocab_size': len(vocab),
    'max_seq_length': max_len,
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'train_unk_rate': train_vec_stats['unk_rate'],
    'val_unk_rate': val_vec_stats['unk_rate'],
    'test_unk_rate': test_vec_stats['unk_rate'],
    'vocab_coverage': vocab_debug['coverage'],
}

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved: config.json")

# === 4. Save vocab_debug.json ===
vocab_debug['force_keep_identifiers'] = list(FORCE_KEEP_IDENTIFIERS)
vocab_debug['preserved_numbers'] = list(PRESERVED_NUMBERS)
vocab_debug['preserved_hex'] = list(PRESERVED_HEX)
vocab_debug['preserved_octal'] = list(PRESERVED_OCTAL)

with open(os.path.join(OUTPUT_DIR, 'vocab_debug.json'), 'w', encoding='utf-8') as f:
    json.dump(vocab_debug, f, indent=2, ensure_ascii=False)
print(f"Saved: vocab_debug.json")

# === 5. Save vectorization_stats.json ===
vec_stats = {
    'train': {
        'shape': list(train_input_ids.shape),
        'unk_rate': train_vec_stats['unk_rate'],
        'total_tokens': train_vec_stats['total_tokens'],
        'total_unks': train_vec_stats['total_unks'],
        'top_unk_tokens': train_vec_stats['top_unk_tokens'][:30],
    },
    'val': {
        'shape': list(val_input_ids.shape),
        'unk_rate': val_vec_stats['unk_rate'],
        'total_tokens': val_vec_stats['total_tokens'],
        'total_unks': val_vec_stats['total_unks'],
        'top_unk_tokens': val_vec_stats['top_unk_tokens'][:30],
    },
    'test': {
        'shape': list(test_input_ids.shape),
        'unk_rate': test_vec_stats['unk_rate'],
        'total_tokens': test_vec_stats['total_tokens'],
        'total_unks': test_vec_stats['total_unks'],
        'top_unk_tokens': test_vec_stats['top_unk_tokens'][:30],
    },
}

with open(os.path.join(OUTPUT_DIR, 'vectorization_stats.json'), 'w', encoding='utf-8') as f:
    json.dump(vec_stats, f, indent=2)
print(f"Saved: vectorization_stats.json")

# %% [markdown]
# ## 10. Save Debug Outputs (tokens.json and token_with_id.json)

# %%
print("\n" + "=" * 60)
print("SAVING DEBUG JSON FILES")
print("=" * 60)

id_to_token = {v: k for k, v in vocab.items()}


def _truncate_text(s: str, limit: int = 500) -> str:
    """Truncate text with ellipsis."""
    s = s or ""
    return (s[:limit] + "...") if len(s) > limit else s


def save_tokens_json(split_name, df, sliced_codes, tokens_list, stats_list, unk_pos_list):
    """Save {split}_tokens.json for a split with ALL samples."""
    tokens_data = {
        'meta': {
            'split': split_name,
            'n_samples': len(df),
            'tokenizer_config': TOKENIZER_CONFIG,
            'vocab_size': len(vocab),
        },
        'samples': []
    }

    for i in tqdm(range(len(df)), desc=f"Building {split_name}_tokens"):
        tokens = tokens_list[i]
        unk_positions = unk_pos_list[i]
        sample = {
            'sample_id': i,
            'label': int(df.iloc[i]['target']),
            'original_code': _truncate_text(df.iloc[i]['func'], 500),
            'sliced_code': _truncate_text(sliced_codes[i], 500),
            'tokens': tokens,
            'n_tokens': len(tokens),
            'n_unk': len(unk_positions),
            'unk_positions': unk_positions,
            'dangerous_apis': stats_list[i].get('dangerous_apis', []),
        }
        tokens_data['samples'].append(sample)

    filepath = os.path.join(OUTPUT_DIR, f'{split_name}_tokens.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tokens_data, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved: {split_name}_tokens.json ({len(df)} samples, {size_mb:.1f} MB)")


def save_token_with_id_json(split_name, df, tokens_list, input_ids_array):
    """Save {split}_token_with_id.json for a split with ALL samples."""
    data = {
        'meta': {
            'split': split_name,
            'vocab_size': len(vocab),
            'max_seq_length': max_len,
        },
        'samples': []
    }

    for i in tqdm(range(len(df)), desc=f"Building {split_name}_token_with_id"):
        tokens_trunc = tokens_list[i][:max_len]
        ids = [int(x) for x in input_ids_array[i].tolist()]
        roundtrip_tokens = [id_to_token.get(int(_id), 'UNK') for _id in ids]

        # Build expected roundtrip for verification
        expected_roundtrip = [
            (tok if tok in vocab else 'UNK') for tok in tokens_trunc
        ] + (['PAD'] * max(0, max_len - len(tokens_trunc)))
        expected_roundtrip = expected_roundtrip[:max_len]

        sample = {
            'sample_id': i,
            'label': int(df.iloc[i]['target']),
            'tokens': tokens_trunc,
            'ids': ids,
            'roundtrip_tokens': roundtrip_tokens,
            'mapping_correct': (roundtrip_tokens == expected_roundtrip),
        }
        data['samples'].append(sample)

    filepath = os.path.join(OUTPUT_DIR, f'{split_name}_token_with_id.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved: {split_name}_token_with_id.json ({len(df)} samples, {size_mb:.1f} MB)")


# Save tokens.json for all splits
save_tokens_json('train', train_df, train_sliced, train_tokens, train_stats, train_unk_pos)
save_tokens_json('val', val_df, val_sliced, val_tokens, val_stats, val_unk_pos)
save_tokens_json('test', test_df, test_sliced, test_tokens, test_stats, test_unk_pos)

# Save token_with_id.json for all splits
save_token_with_id_json('train', train_df, train_tokens, train_input_ids)
save_token_with_id_json('val', val_df, val_tokens, val_input_ids)
save_token_with_id_json('test', test_df, test_tokens, test_input_ids)

# %% [markdown]
# ## 11. Verify Outputs

# %%
print("\n" + "=" * 60)
print("OUTPUT VERIFICATION")
print("=" * 60)

# List output files
output_files = list(Path(OUTPUT_DIR).glob('*'))
print(f"\nOutput files ({len(output_files)}):")
for f in sorted(output_files):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {size_mb:.2f} MB")

# Verify data shapes
print("\n--- Data Shapes ---")
for split in ['train', 'val', 'test']:
    data = np.load(os.path.join(OUTPUT_DIR, f'{split}.npz'))
    print(f"{split}: input_ids={data['input_ids'].shape}, labels={data['labels'].shape}")

# Verify vocab
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'r') as f:
    loaded_vocab = json.load(f)
print(f"\nVocab size: {len(loaded_vocab)}")

# Sample token-to-ID mapping check
print("\n--- Sample Token-to-ID Verification ---")
sample_tokens = ['PAD', 'UNK', 'SEP', 'malloc', 'strcpy', 'buf', 'len', '0', '1', 'NEG_1', 'NUM']
for tok in sample_tokens:
    if tok in loaded_vocab:
        print(f"  {tok}: {loaded_vocab[tok]}")
    else:
        print(f"  {tok}: NOT IN VOCAB")

print("\n" + "=" * 60)
print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"Ready for training with:")
print(f"  - train.npz: {train_input_ids.shape[0]} samples")
print(f"  - val.npz: {val_input_ids.shape[0]} samples")
print(f"  - test.npz: {test_input_ids.shape[0]} samples")

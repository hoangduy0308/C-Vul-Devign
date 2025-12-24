# %% [markdown]
# # Full Pipeline: Devign Vulnerability Detection (Unified)
# 
# Merged from prepare_data.py + 03_preprocessing_v2.py:
# 1. **Load**: Raw Devign dataset from parquet files
# 2. **Dedup**: Check & remove duplicates (prevent data leakage)
# 3. **Slice**: PDG-based slicing (backward + forward dependencies)
# 4. **Features**: Extract 22 vulnerability features from slices
# 5. **Tokenize**: Configurable (PreserveIdentifierTokenizer or CanonicalTokenizer)
# 6. **Vectorize**: Build vocab, convert to input_ids with attention masks
# 7. **Save**: NPZ files for training, JSON/JSONL for debugging
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
    sys.path.insert(0, '/kaggle/input/devign-pipeline')
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

# Check GPU
try:
    import torch
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("PyTorch not available")

# %% [markdown]
# ## 2. Configuration

# %%
# ============================================================
# MAIN CONFIGURATION - Edit these values as needed
# ============================================================

# Tokenizer choice: 'preserve' or 'canonical'
# - 'preserve': PreserveIdentifierTokenizer (vocab 30k, keeps variable names)
# - 'canonical': CanonicalTokenizer (vocab 500, semantic buckets BUF_k/LEN_k/etc.)
TOKENIZER_TYPE = 'preserve'

# PDG-based slicing configuration
# Tuned based on SySeVR/DeepWukong unbounded traversal approach but with budget limits
PDG_SLICE_CONFIG = {
    'backward_depth': 3,             # Max 3 hops backward (follows actual deps)
    'forward_depth': 2,              # Max 2 hops forward
    'include_data_deps': True,       # Include data dependencies
    'include_control_deps': True,    # Include control dependencies
    'control_predicate_only': True,  # Only include if/while headers, not full blocks
    'max_lines': 30,                 # Hard cap on output lines
    'max_tokens': 256,               # Hard cap on tokens
    'fallback_window': 3,            # Small fallback if PDG fails
}

# Tokenization configuration
if TOKENIZER_TYPE == 'preserve':
    TOKENIZER_CONFIG = {
        'min_freq': 2,
        'max_vocab_size': 30000,
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
else:  # canonical
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
    'min_slice_tokens': 10,  # Fallback to full code if slice too short
    'n_jobs': 1,
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'extract_features': True,  # Set False to skip feature extraction
    'normalize_features': True,
}

# Debug output configuration
DEBUG_CONFIG = {
    'save_tokens_jsonl': True,  # Save token debug files
    'save_token_with_id_jsonl': True,
}

print("=== Configuration ===")
print(f"Tokenizer type: {TOKENIZER_TYPE}")
print(f"PDG slice config: {json.dumps(PDG_SLICE_CONFIG, indent=2)}")
print(f"Tokenizer config: {json.dumps(TOKENIZER_CONFIG, indent=2)}")

# %% [markdown]
# ## 3. Import Modules

# %%
from src.data.loader import DevignLoader
from src.slicing.pdg_slicer import PDGSlicer, PDGSliceConfig, PDGSliceType
from src.slicing.utils import find_criterion_lines
from src.tokenization.hybrid_tokenizer import DANGEROUS_APIS, SPECIAL_TOKENS

if TOKENIZER_TYPE == 'preserve':
    from src.tokenization.preserve_tokenizer import (
        PreserveIdentifierTokenizer,
        build_preserve_vocab,
        vectorize_batch_preserve,
        FORCE_KEEP_IDENTIFIERS,
        PRESERVED_NUMBERS,
        PRESERVED_HEX,
        PRESERVED_OCTAL,
    )
else:
    from src.tokenization.hybrid_tokenizer import (
        CanonicalTokenizer,
        build_hybrid_vocab,
        vectorize,
        get_canonical_vocab_tokens
    )

if FEATURE_CONFIG['extract_features']:
    from src.vuln.slice_features import extract_slice_features, extract_slice_features_batch, SLICE_FEATURE_NAMES
    from src.vuln.dictionary import VulnDictionary

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

# Validate columns exist and drop rows with NaN/empty/whitespace-only func
for split_name in ['train', 'val', 'test']:
    df = {'train': train_df, 'val': val_df, 'test': test_df}[split_name]
    assert 'func' in df.columns, f"Missing 'func' column in {split_name}"
    assert 'target' in df.columns, f"Missing 'target' column in {split_name}"
    
    original_count = len(df)
    nan_count = df['func'].isna().sum()
    empty_count = (df['func'].fillna('') == '').sum() - nan_count
    whitespace_count = df['func'].fillna('').str.strip().eq('').sum() - nan_count - empty_count
    
    # Drop rows with NaN, empty, or whitespace-only func
    valid_mask = df['func'].notna() & (df['func'].str.strip() != '')
    df_clean = df[valid_mask].reset_index(drop=True)
    
    dropped_count = original_count - len(df_clean)
    if dropped_count > 0:
        print(f"Warning: {split_name} dropped {dropped_count} rows (NaN: {nan_count}, empty: {empty_count}, whitespace-only: {whitespace_count})")
    
    # Update the dataframe
    if split_name == 'train':
        train_df = df_clean
    elif split_name == 'val':
        val_df = df_clean
    else:
        test_df = df_clean

print(f"\nDataset loaded:")
print(f"  Train: {len(train_df)} samples (vuln: {(train_df['target']==1).sum()})")
print(f"  Val: {len(val_df)} samples (vuln: {(val_df['target']==1).sum()})")
print(f"  Test: {len(test_df)} samples (vuln: {(test_df['target']==1).sum()})")

# %% [markdown]
# ## 4.1. Check & Remove Duplicates (Data Leakage Prevention)

# %%
print("\n" + "=" * 60)
print("STEP 1.1: DATA LEAKAGE CHECK & DUPLICATE REMOVAL")
print("=" * 60)

# Check for duplicates within each split
train_dups_internal = train_df['func'].duplicated().sum()
val_dups_internal = val_df['func'].duplicated().sum()
test_dups_internal = test_df['func'].duplicated().sum()

print(f"\nInternal duplicates:")
print(f"  Train: {train_dups_internal} duplicates")
print(f"  Val: {val_dups_internal} duplicates")
print(f"  Test: {test_dups_internal} duplicates")

# Check for cross-split duplicates (DATA LEAKAGE!)
train_funcs = set(train_df['func'].tolist())
val_funcs = set(val_df['func'].tolist())
test_funcs = set(test_df['func'].tolist())

train_val_overlap = train_funcs & val_funcs
train_test_overlap = train_funcs & test_funcs
val_test_overlap = val_funcs & test_funcs

print(f"\nCross-split duplicates (DATA LEAKAGE):")
print(f"  Train-Val overlap: {len(train_val_overlap)} functions")
print(f"  Train-Test overlap: {len(train_test_overlap)} functions")
print(f"  Val-Test overlap: {len(val_test_overlap)} functions")

# Remove cross-split duplicates: keep in train, remove from val/test
if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
    print("\n[WARNING] Removing cross-split duplicates to prevent data leakage...")
    
    val_before = len(val_df)
    test_before = len(test_df)
    
    # Remove train overlaps from val/test
    val_df = val_df[~val_df['func'].isin(train_val_overlap)].reset_index(drop=True)
    test_df = test_df[~test_df['func'].isin(train_test_overlap)].reset_index(drop=True)
    
    # Remove val-test overlap from test (keep in val, remove from test)
    test_df = test_df[~test_df['func'].isin(val_test_overlap)].reset_index(drop=True)
    
    print(f"  Val: {val_before} -> {len(val_df)} (removed {val_before - len(val_df)} train overlaps)")
    print(f"  Test: {test_before} -> {len(test_df)} (removed train + val overlaps)")

# Remove internal duplicates within each split
print("\nRemoving internal duplicates...")
train_before = len(train_df)
val_before = len(val_df)
test_before = len(test_df)

train_df = train_df.drop_duplicates(subset='func', keep='first').reset_index(drop=True)
val_df = val_df.drop_duplicates(subset='func', keep='first').reset_index(drop=True)
test_df = test_df.drop_duplicates(subset='func', keep='first').reset_index(drop=True)

print(f"  Train: {train_before} -> {len(train_df)} (removed {train_before - len(train_df)})")
print(f"  Val: {val_before} -> {len(val_df)} (removed {val_before - len(val_df)})")
print(f"  Test: {test_before} -> {len(test_df)} (removed {test_before - len(test_df)})")

print(f"\n✓ After deduplication:")
print(f"  Train: {len(train_df)} samples (vuln: {(train_df['target']==1).sum()})")
print(f"  Val: {len(val_df)} samples (vuln: {(val_df['target']==1).sum()})")
print(f"  Test: {len(test_df)} samples (vuln: {(test_df['target']==1).sum()})")

# %% [markdown]
# ## 5. PDG-based Slicing

# %%
print("\n" + "=" * 60)
print("STEP 2: PDG-BASED SLICING")
print("=" * 60)

# Initialize PDG-based slicer
pdg_config = PDGSliceConfig(
    slice_type=PDGSliceType.BIDIRECTIONAL,
    backward_depth=PDG_SLICE_CONFIG['backward_depth'],
    forward_depth=PDG_SLICE_CONFIG['forward_depth'],
    include_data_deps=PDG_SLICE_CONFIG['include_data_deps'],
    include_control_deps=PDG_SLICE_CONFIG['include_control_deps'],
    control_predicate_only=PDG_SLICE_CONFIG['control_predicate_only'],
    max_lines=PDG_SLICE_CONFIG['max_lines'],
    max_tokens=PDG_SLICE_CONFIG['max_tokens'],
    fallback_window=PDG_SLICE_CONFIG['fallback_window'],
)
slicer = PDGSlicer(pdg_config)


def estimate_tokens(code: str) -> int:
    """Estimate token count by counting words, operators, and punctuation.
    
    Counts:
    - Words/identifiers (e.g., 'malloc', 'buf', 'i')
    - Multi-char operators (e.g., '==', '!=', '->', '<<')
    - Single-char operators and punctuation (e.g., '+', ';', '{')
    """
    # Multi-char operators (order matters - check longer first)
    multi_ops = re.findall(r'->|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=', code)
    # Remove multi-char operators to avoid double counting
    code_clean = re.sub(r'->|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=', ' ', code)
    # Words/identifiers and numbers
    words = re.findall(r'\b\w+\b', code_clean)
    # Single-char operators and punctuation
    single_ops = re.findall(r'[+\-*/%&|^~!<>=;:,.\[\]{}()#?]', code_clean)
    return len(words) + len(multi_ops) + len(single_ops)


def insert_sep_in_middle(code: str, sep_token: str = "[SEP]") -> str:
    """Insert SEP token in the middle of code for fallback cases."""
    lines = code.strip().split('\n')
    if len(lines) <= 1:
        return f"{code.strip()} {sep_token}"
    mid = len(lines) // 2
    before = '\n'.join(lines[:mid])
    after = '\n'.join(lines[mid:])
    return f"{before} {sep_token} {after}"


def process_pdg_slice_batch(df, batch_size=500):
    """Process PDG-based slicing in batches with fallback for short slices."""
    from collections import defaultdict
    
    codes = df['func'].tolist()
    n_samples = len(codes)
    
    if n_samples == 0:
        print("  [WARNING] Empty dataset, skipping slicing")
        return [], []
    
    sliced_codes = []
    fallback_count = 0
    pdg_success_count = 0
    exception_counts = defaultdict(int)
    
    for start in tqdm(range(0, n_samples, batch_size), desc="PDG Slicing"):
        end = min(start + batch_size, n_samples)
        batch_codes = codes[start:end]
        batch_criteria = [find_criterion_lines(code) for code in batch_codes]
        
        for code, criteria in zip(batch_codes, batch_criteria):
            try:
                result = slicer.slice(code, criteria)
                sliced = result.code
                
                # Track PDG vs fallback usage
                if not result.used_fallback:
                    pdg_success_count += 1
                
                # Check if slice is too short - fallback to full function with SEP
                if estimate_tokens(sliced) < PROCESS_CONFIG['min_slice_tokens']:
                    sliced = insert_sep_in_middle(code)
                    fallback_count += 1
                
                sliced_codes.append(sliced)
            except Exception as e:
                sliced_codes.append(insert_sep_in_middle(code))
                fallback_count += 1
                exception_counts[type(e).__name__] += 1
    
    print(f"  PDG success: {pdg_success_count}/{n_samples} ({100*pdg_success_count/n_samples:.1f}%)")
    print(f"  Fallback to full function: {fallback_count} samples")
    if exception_counts:
        print(f"  Exceptions summary:")
        for exc_type, count in sorted(exception_counts.items(), key=lambda x: -x[1]):
            print(f"    {exc_type}: {count}")
    return sliced_codes


# Process all splits
print("\nProcessing train set...")
train_sliced = process_pdg_slice_batch(train_df, PROCESS_CONFIG['batch_size'])

print("\nProcessing val set...")
val_sliced = process_pdg_slice_batch(val_df, PROCESS_CONFIG['batch_size'])

print("\nProcessing test set...")
test_sliced = process_pdg_slice_batch(test_df, PROCESS_CONFIG['batch_size'])

print(f"\nPDG slicing completed:")
print(f"  Train: {len(train_sliced)} samples")
print(f"  Val: {len(val_sliced)} samples")
print(f"  Test: {len(test_sliced)} samples")

# Slice length statistics
if train_sliced:
    train_slice_lens = [len(s.split('\n')) for s in train_sliced]
    print(f"\nTrain slice length stats (lines):")
    print(f"  Mean: {np.mean(train_slice_lens):.1f}")
    print(f"  Median: {np.median(train_slice_lens):.1f}")
    print(f"  Max: {max(train_slice_lens)}")
    print(f"  Min: {min(train_slice_lens)}")

# %% [markdown]
# ## 6. Extract Vulnerability Features (Optional)

# %%
if FEATURE_CONFIG['extract_features']:
    print("\n" + "=" * 60)
    print("STEP 3: EXTRACTING VULNERABILITY FEATURES")
    print("=" * 60)
    
    vuln_dict = VulnDictionary()
    
    print(f"Feature count: {len(SLICE_FEATURE_NAMES)}")
    print(f"Features: {SLICE_FEATURE_NAMES}")
    
    def process_features_batch(sliced_codes, original_codes, batch_size=500):
        """Extract features in batches."""
        n_samples = len(sliced_codes)
        all_features = []
        
        for start in tqdm(range(0, n_samples, batch_size), desc="Extracting features"):
            end = min(start + batch_size, n_samples)
            batch_slices = sliced_codes[start:end]
            batch_fulls = original_codes[start:end]
            
            batch_features = extract_slice_features_batch(
                batch_slices, batch_fulls, vuln_dict, n_jobs=1
            )
            all_features.extend(batch_features)
        
        return all_features
    
    print("\nExtracting train features...")
    train_features = process_features_batch(train_sliced, train_df['func'].tolist(), PROCESS_CONFIG['batch_size'])
    
    print("Extracting val features...")
    val_features = process_features_batch(val_sliced, val_df['func'].tolist(), PROCESS_CONFIG['batch_size'])
    
    print("Extracting test features...")
    test_features = process_features_batch(test_sliced, test_df['func'].tolist(), PROCESS_CONFIG['batch_size'])
    
    print(f"\nFeature extraction completed:")
    print(f"  Train: {len(train_features)} samples x {len(SLICE_FEATURE_NAMES)} features")
    print(f"  Val: {len(val_features)} samples")
    print(f"  Test: {len(test_features)} samples")
else:
    train_features = val_features = test_features = None
    print("\n[SKIP] Feature extraction disabled")

# %% [markdown]
# ## 7. Tokenization

# %%
print("\n" + "=" * 60)
print("STEP 4: TOKENIZATION")
print("=" * 60)

if TOKENIZER_TYPE == 'preserve':
    tokenizer = PreserveIdentifierTokenizer(TOKENIZER_CONFIG)
    
    print("\nTokenizing train set...")
    train_tokens, train_details, train_stats = tokenizer.tokenize_batch(train_sliced, with_details=True)
    
    print("Tokenizing val set...")
    val_tokens, val_details, val_stats = tokenizer.tokenize_batch(val_sliced, with_details=True)
    
    print("Tokenizing test set...")
    test_tokens, test_details, test_stats = tokenizer.tokenize_batch(test_sliced, with_details=True)
else:
    tokenizer = CanonicalTokenizer(
        preserve_dangerous_apis=TOKENIZER_CONFIG['preserve_dangerous_apis'],
        preserve_keywords=TOKENIZER_CONFIG['preserve_keywords']
    )
    
    def tokenize_batch_canonical(codes, batch_size=500):
        """Tokenize codes in batches."""
        n_samples = len(codes)
        all_tokens = []
        
        for start in tqdm(range(0, n_samples, batch_size), desc="Tokenizing"):
            end = min(start + batch_size, n_samples)
            batch_codes = codes[start:end]
            batch_tokens = tokenizer.tokenize_batch(batch_codes, n_jobs=1)
            all_tokens.extend(batch_tokens)
        
        return all_tokens
    
    print("\nTokenizing train set...")
    train_tokens = tokenize_batch_canonical(train_sliced, PROCESS_CONFIG['batch_size'])
    
    print("Tokenizing val set...")
    val_tokens = tokenize_batch_canonical(val_sliced, PROCESS_CONFIG['batch_size'])
    
    print("Tokenizing test set...")
    test_tokens = tokenize_batch_canonical(test_sliced, PROCESS_CONFIG['batch_size'])
    
    # Placeholder stats for canonical tokenizer (correct lengths per split)
    train_stats = [{} for _ in range(len(train_tokens))]
    val_stats = [{} for _ in range(len(val_tokens))]
    test_stats = [{} for _ in range(len(test_tokens))]
    train_details = val_details = test_details = None

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
# ## 8. Build Vocabulary

# %%
print("\n" + "=" * 60)
print("STEP 5: BUILD VOCABULARY")
print("=" * 60)

if TOKENIZER_TYPE == 'preserve':
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
    
    print(f"\n✓ Critical Tokens Verification:")
    print(f"  Dangerous APIs in vocab: {vocab_debug['dangerous_apis_in_vocab']}/{vocab_debug['dangerous_apis_total']}")
    print(f"  C Keywords in vocab: {vocab_debug['keywords_in_vocab']}/{vocab_debug['keywords_total']}")
else:
    # Build vocab from train_tokens (already tokenized) for canonical tokenizer
    from collections import Counter
    token_counts = Counter()
    for tokens in train_tokens:
        token_counts.update(tokens)
    
    # Start with special tokens
    vocab = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'SEP': 4}
    next_id = len(vocab)
    
    # Add dangerous APIs first
    for api in DANGEROUS_APIS:
        if api not in vocab:
            vocab[api] = next_id
            next_id += 1
    
    # Add remaining tokens by frequency
    for tok, count in token_counts.most_common():
        if tok not in vocab and count >= TOKENIZER_CONFIG['min_freq']:
            if len(vocab) >= TOKENIZER_CONFIG['max_vocab_size']:
                break
            vocab[tok] = next_id
            next_id += 1
    
    vocab_debug = {'vocab_size': len(vocab), 'coverage': 1.0}
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample vocab entries: {dict(list(vocab.items())[:15])}")

# Verify important tokens
print(f"\nVerification:")
important_tokens = ['PAD', 'UNK', 'SEP', 'malloc', 'strcpy', 'free']
for tok in important_tokens:
    status = f"ID={vocab[tok]}" if tok in vocab else "NOT IN VOCAB"
    print(f"  '{tok}': {status}")

# Verify vocab is built ONLY from training data
print(f"\n✓ Data Leakage Check:")
print(f"  Vocab built from: TRAINING DATA ONLY ({len(train_tokens)} samples)")
print(f"  Val/Test tokens not used for vocab building: ✓")

# %% [markdown]
# ## 9. Vectorize Data

# %%
print("\n" + "=" * 60)
print("STEP 6: VECTORIZATION")
print("=" * 60)

max_len = TOKENIZER_CONFIG['max_seq_length']

if TOKENIZER_TYPE == 'preserve':
    print("\nVectorizing train set...")
    train_input_ids, train_attention_mask, train_unk_pos, train_vec_stats = vectorize_batch_preserve(
        train_tokens, vocab, max_len
    )
    
    print("Vectorizing val set...")
    val_input_ids, val_attention_mask, val_unk_pos, val_vec_stats = vectorize_batch_preserve(
        val_tokens, vocab, max_len
    )
    
    print("Vectorizing test set...")
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
else:
    def vectorize_batch_hybrid(tokens_list, vocab, max_len, batch_size=500):
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
    
    print("\nVectorizing train set...")
    train_input_ids, train_attention_mask = vectorize_batch_hybrid(train_tokens, vocab, max_len, PROCESS_CONFIG['batch_size'])
    
    print("Vectorizing val set...")
    val_input_ids, val_attention_mask = vectorize_batch_hybrid(val_tokens, vocab, max_len, PROCESS_CONFIG['batch_size'])
    
    print("Vectorizing test set...")
    test_input_ids, test_attention_mask = vectorize_batch_hybrid(test_tokens, vocab, max_len, PROCESS_CONFIG['batch_size'])
    
    train_unk_pos = [[] for _ in range(len(train_tokens))]
    val_unk_pos = [[] for _ in range(len(val_tokens))]
    test_unk_pos = [[] for _ in range(len(test_tokens))]
    
    def compute_unk_stats(input_ids_array, tokens_list, unk_id=1):
        """Compute UNK statistics from vectorized input_ids.
        
        Args:
            input_ids_array: numpy array of shape (n_samples, max_len)
            tokens_list: list of token lists (before vectorization)
            unk_id: vocab ID for UNK token (default: 1)
        
        Returns:
            dict with unk_rate, top_unk_tokens, total_tokens, total_unks
        """
        from collections import Counter
        unk_counter = Counter()
        total_tokens = 0
        total_unks = 0
        
        for i, tokens in enumerate(tokens_list):
            ids = input_ids_array[i]
            # Only count actual tokens (not padding)
            seq_len = min(len(tokens), len(ids))
            for j in range(seq_len):
                total_tokens += 1
                if ids[j] == unk_id:
                    total_unks += 1
                    if j < len(tokens):
                        unk_counter[tokens[j]] += 1
        
        unk_rate = total_unks / total_tokens if total_tokens > 0 else 0
        top_unk_tokens = unk_counter.most_common(50)
        
        return {
            'unk_rate': unk_rate,
            'top_unk_tokens': top_unk_tokens,
            'total_tokens': total_tokens,
            'total_unks': total_unks
        }
    
    train_vec_stats = compute_unk_stats(train_input_ids, train_tokens)
    val_vec_stats = compute_unk_stats(val_input_ids, val_tokens)
    test_vec_stats = compute_unk_stats(test_input_ids, test_tokens)
    
    print(f"\nVectorization completed:")
    print(f"  Train: {train_input_ids.shape}, UNK rate: {train_vec_stats['unk_rate']:.2%}")
    print(f"  Val: {val_input_ids.shape}, UNK rate: {val_vec_stats['unk_rate']:.2%}")
    print(f"  Test: {test_input_ids.shape}, UNK rate: {test_vec_stats['unk_rate']:.2%}")
    
    if train_vec_stats['top_unk_tokens']:
        print(f"\nTop UNK tokens (train):")
        for tok, count in train_vec_stats['top_unk_tokens'][:15]:
            print(f"  {tok}: {count}")

# %% [markdown]
# ## 10. Process Vulnerability Features

# %%
if FEATURE_CONFIG['extract_features'] and train_features:
    print("\n" + "=" * 60)
    print("STEP 7: PROCESSING VULNERABILITY FEATURES")
    print("=" * 60)
    
    def features_to_array(features_list):
        """Convert list of feature dicts to numpy array."""
        n_samples = len(features_list)
        n_features = len(SLICE_FEATURE_NAMES)
        
        arr = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i, feat_dict in enumerate(features_list):
            for j, name in enumerate(SLICE_FEATURE_NAMES):
                arr[i, j] = feat_dict.get(name, 0.0)
        
        return arr
    
    def normalize_features(train_feat, val_feat, test_feat, feature_names):
        """Normalize features using log1p + z-score."""
        count_features = [
            'loc_slice', 'stmt_count_slice', 'dangerous_call_count_slice',
            'dangerous_call_without_check_count_slice', 'pointer_deref_count_slice',
            'pointer_deref_without_null_check_count_slice', 'array_access_count_slice',
            'array_access_without_bounds_check_count_slice', 'null_check_count_slice',
            'bounds_check_count_slice', 'loc_full'
        ]
        
        feature_names_list = list(feature_names)
        
        train_norm = train_feat.copy()
        val_norm = val_feat.copy()
        test_norm = test_feat.copy()
        
        for feat_name in count_features:
            if feat_name in feature_names_list:
                idx = feature_names_list.index(feat_name)
                train_norm[:, idx] = np.log1p(train_norm[:, idx])
                val_norm[:, idx] = np.log1p(val_norm[:, idx])
                test_norm[:, idx] = np.log1p(test_norm[:, idx])
        
        train_mean = np.mean(train_norm, axis=0)
        train_std = np.std(train_norm, axis=0)
        train_std[train_std == 0] = 1.0
        
        train_norm = (train_norm - train_mean) / train_std
        val_norm = (val_norm - train_mean) / train_std
        test_norm = (test_norm - train_mean) / train_std
        
        norm_stats = {
            'mean': train_mean.tolist(),
            'std': train_std.tolist(),
            'count_features': count_features,
            'feature_names': feature_names_list
        }
        
        return train_norm, val_norm, test_norm, norm_stats
    
    # Convert features to arrays
    print("Converting features to arrays...")
    train_vuln_features = features_to_array(train_features)
    val_vuln_features = features_to_array(val_features)
    test_vuln_features = features_to_array(test_features)
    
    print(f"  Train: {train_vuln_features.shape}")
    print(f"  Val: {val_vuln_features.shape}")
    print(f"  Test: {test_vuln_features.shape}")
    
    if FEATURE_CONFIG['normalize_features']:
        print("\nNormalizing features (log1p + z-score)...")
        train_vuln_features, val_vuln_features, test_vuln_features, norm_stats = normalize_features(
            train_vuln_features, val_vuln_features, test_vuln_features, SLICE_FEATURE_NAMES
        )
        
        print("\nFeature stats after normalization (train):")
        for i, name in enumerate(SLICE_FEATURE_NAMES[:5]):
            col = train_vuln_features[:, i]
            print(f"  {name}: mean={np.mean(col):.3f}, std={np.std(col):.3f}")
    else:
        norm_stats = None
else:
    train_vuln_features = val_vuln_features = test_vuln_features = None
    norm_stats = None

# %% [markdown]
# ## 11. Save Outputs

# %%
print("\n" + "=" * 60)
print("STEP 8: SAVING OUTPUTS")
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

# === 2. Save vulnerability features (if extracted) ===
if train_vuln_features is not None:
    print("\nSaving vulnerability features...")
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
    print(f"Saved: train_vuln.npz, val_vuln.npz, test_vuln.npz")
    
    if norm_stats:
        with open(os.path.join(OUTPUT_DIR, 'feature_norm_stats.json'), 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print("Saved: feature_norm_stats.json")

# === 3. Save vocab.json ===
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
    json.dump(vocab, f, indent=2, ensure_ascii=False)
print(f"Saved: vocab.json ({len(vocab)} tokens)")

# === 4. Save config.json ===
config = {
    'tokenizer_type': TOKENIZER_TYPE,
    'pdg_slice_config': PDG_SLICE_CONFIG,
    'tokenizer_config': TOKENIZER_CONFIG,
    'vocab_size': len(vocab),
    'max_seq_length': max_len,
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'features_extracted': FEATURE_CONFIG['extract_features'],
    'n_features': len(SLICE_FEATURE_NAMES) if FEATURE_CONFIG['extract_features'] else 0,
    'feature_names': SLICE_FEATURE_NAMES if FEATURE_CONFIG['extract_features'] else [],
    'slicing_method': f"PDG-based (backward_depth={PDG_SLICE_CONFIG['backward_depth']}, forward_depth={PDG_SLICE_CONFIG['forward_depth']})",
}

if TOKENIZER_TYPE == 'preserve':
    config['train_unk_rate'] = train_vec_stats['unk_rate']
    config['val_unk_rate'] = val_vec_stats['unk_rate']
    config['test_unk_rate'] = test_vec_stats['unk_rate']
    config['vocab_coverage'] = vocab_debug.get('coverage', 1.0)

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"Saved: config.json")

# === 5. Save vocab_debug.json (preserve tokenizer only) ===
if TOKENIZER_TYPE == 'preserve':
    vocab_debug['force_keep_identifiers'] = list(FORCE_KEEP_IDENTIFIERS)
    vocab_debug['preserved_numbers'] = list(PRESERVED_NUMBERS)
    vocab_debug['preserved_hex'] = list(PRESERVED_HEX)
    vocab_debug['preserved_octal'] = list(PRESERVED_OCTAL)
    
    with open(os.path.join(OUTPUT_DIR, 'vocab_debug.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_debug, f, indent=2, ensure_ascii=False)
    print(f"Saved: vocab_debug.json")
    
    # === 6. Save vectorization_stats.json ===
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
# ## 12. Save Debug Outputs (JSONL)

# %%
if DEBUG_CONFIG['save_tokens_jsonl'] and TOKENIZER_TYPE == 'preserve':
    print("\n" + "=" * 60)
    print("SAVING DEBUG JSON FILES")
    print("=" * 60)
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    def _truncate_text(s: str, limit: int = 500) -> str:
        """Truncate text with ellipsis."""
        s = s or ""
        return (s[:limit] + "...") if len(s) > limit else s
    
    def save_tokens_jsonl(split_name, df, sliced_codes, tokens_list, stats_list, unk_pos_list):
        """Save {split}_tokens.jsonl in JSONL format."""
        meta_filepath = os.path.join(OUTPUT_DIR, f'{split_name}_tokens_meta.json')
        meta = {
            'split': split_name,
            'n_samples': len(df),
            'tokenizer_config': TOKENIZER_CONFIG,
            'vocab_size': len(vocab),
        }
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        filepath = os.path.join(OUTPUT_DIR, f'{split_name}_tokens.jsonl')
        with open(filepath, 'w', encoding='utf-8') as f:
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
                    'dangerous_apis': stats_list[i].get('dangerous_apis', []) if stats_list[i] else [],
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Saved: {split_name}_tokens.jsonl ({len(df)} samples, {size_mb:.1f} MB)")
    
    def save_token_with_id_jsonl(split_name, df, tokens_list, input_ids_array):
        """Save {split}_token_with_id.jsonl in JSONL format."""
        meta_filepath = os.path.join(OUTPUT_DIR, f'{split_name}_token_with_id_meta.json')
        meta = {
            'split': split_name,
            'vocab_size': len(vocab),
            'max_seq_length': max_len,
        }
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        filepath = os.path.join(OUTPUT_DIR, f'{split_name}_token_with_id.jsonl')
        with open(filepath, 'w', encoding='utf-8') as f:
            for i in tqdm(range(len(df)), desc=f"Building {split_name}_token_with_id"):
                tokens_trunc = tokens_list[i][:max_len]
                ids = [int(x) for x in input_ids_array[i].tolist()]
                roundtrip_tokens = [id_to_token.get(int(_id), 'UNK') for _id in ids]
                
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
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Saved: {split_name}_token_with_id.jsonl ({len(df)} samples, {size_mb:.1f} MB)")
    
    # Save tokens.jsonl for all splits
    save_tokens_jsonl('train', train_df, train_sliced, train_tokens, train_stats, train_unk_pos)
    save_tokens_jsonl('val', val_df, val_sliced, val_tokens, val_stats, val_unk_pos)
    save_tokens_jsonl('test', test_df, test_sliced, test_tokens, test_stats, test_unk_pos)
    
    # Save token_with_id.jsonl for all splits
    if DEBUG_CONFIG['save_token_with_id_jsonl']:
        save_token_with_id_jsonl('train', train_df, train_tokens, train_input_ids)
        save_token_with_id_jsonl('val', val_df, val_tokens, val_input_ids)
        save_token_with_id_jsonl('test', test_df, test_tokens, test_input_ids)

# %% [markdown]
# ## 13. Verify Outputs

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
    print(f"  Label distribution: 0={np.sum(data['labels']==0)}, 1={np.sum(data['labels']==1)}")

# Verify vulnerability features
if train_vuln_features is not None:
    print("\n--- Vulnerability Features ---")
    for split in ['train', 'val', 'test']:
        data = np.load(os.path.join(OUTPUT_DIR, f'{split}_vuln.npz'), allow_pickle=True)
        features = data['features']
        print(f"{split}_vuln: {features.shape}")

# Verify vocab
with open(os.path.join(OUTPUT_DIR, 'vocab.json'), 'r') as f:
    loaded_vocab = json.load(f)
print(f"\nVocab size: {len(loaded_vocab)}")

# Sample token-to-ID mapping check
print("\n--- Sample Token-to-ID Verification ---")
sample_tokens = ['PAD', 'UNK', 'SEP', 'malloc', 'strcpy', 'buf', 'len', '0', '1']
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
if train_vuln_features is not None:
    print(f"  - Vulnerability features: {train_vuln_features.shape[1]} features per sample")

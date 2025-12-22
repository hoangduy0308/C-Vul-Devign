"""
Comprehensive Debug Script for Devign Preprocessing V2
Analyzes output in Dataset/devign_slice_v2/
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import sys

# Add pipeline to path
sys.path.insert(0, 'F:/Work/C Vul Devign/devign_pipeline')

OUTPUT_DIR = Path('F:/Work/C Vul Devign/Dataset/devign_slice_v2')

def load_data():
    """Load all processed data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load vocab
    with open(OUTPUT_DIR / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    print(f"Vocab size: {len(vocab)}")
    
    # Load config
    with open(OUTPUT_DIR / 'config.json', 'r') as f:
        config = json.load(f)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Load splits
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


def analyze_vocab(vocab):
    """Analyze vocabulary composition."""
    print("\n" + "=" * 70)
    print("VOCABULARY ANALYSIS")
    print("=" * 70)
    
    # Reverse vocab (id -> token)
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Categorize tokens
    special_tokens = ['PAD', 'UNK', 'BOS', 'EOS', 'SEP']
    semantic_buckets = ['BUF_', 'LEN_', 'PTR_', 'IDX_', 'SENS_', 'PRIV_', 'VAR_']
    
    special_count = sum(1 for t in vocab if t in special_tokens)
    
    bucket_counts = {}
    for prefix in semantic_buckets:
        bucket_counts[prefix.rstrip('_')] = sum(1 for t in vocab if t.startswith(prefix))
    
    # Dangerous APIs
    from src.tokenization.hybrid_tokenizer import DANGEROUS_APIS, C_KEYWORDS
    apis_in_vocab = [t for t in vocab if t in DANGEROUS_APIS]
    keywords_in_vocab = [t for t in vocab if t in C_KEYWORDS]
    
    # Operators and punctuation
    operators = [t for t in vocab if t in '+-*/%&|^~<>=!']
    multi_char_ops = [t for t in vocab if t in ['==', '!=', '<=', '>=', '&&', '||', '->', '<<', '>>']]
    punctuation = [t for t in vocab if t in '(){}[];,.:?#']
    
    print(f"\n[Token Categories]")
    print(f"  Special tokens: {special_count}")
    print(f"  Semantic buckets:")
    for bucket, count in bucket_counts.items():
        print(f"    {bucket}: {count}")
    print(f"  Dangerous APIs: {len(apis_in_vocab)}")
    print(f"  C Keywords: {len(keywords_in_vocab)}")
    print(f"  Operators: {len(operators) + len(multi_char_ops)}")
    print(f"  Punctuation: {len(punctuation)}")
    
    print(f"\n[Dangerous APIs in Vocab ({len(apis_in_vocab)}):]")
    print(f"  {sorted(apis_in_vocab)}")
    
    print(f"\n[C Keywords in Vocab ({len(keywords_in_vocab)}):]")
    print(f"  {sorted(keywords_in_vocab)}")
    
    # Check for SEP token
    if 'SEP' in vocab:
        print(f"\n[OK] SEP token present: vocab['SEP'] = {vocab['SEP']}")
    else:
        print(f"\n[ERROR] SEP token NOT in vocab!")
    
    return apis_in_vocab, keywords_in_vocab, bucket_counts


def analyze_sequences(data, vocab):
    """Analyze tokenized sequences."""
    print("\n" + "=" * 70)
    print("SEQUENCE ANALYSIS")
    print("=" * 70)
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    for split, split_data in data.items():
        input_ids = split_data['input_ids']
        attention_mask = split_data['attention_mask']
        labels = split_data['labels']
        
        print(f"\n[{split.upper()}]")
        print(f"  Shape: {input_ids.shape}")
        
        # Token length analysis (non-padding)
        lengths = attention_mask.sum(axis=1)
        print(f"  Token lengths:")
        print(f"    Mean: {np.mean(lengths):.1f}")
        print(f"    Median: {np.median(lengths):.1f}")
        print(f"    Min: {np.min(lengths)}, Max: {np.max(lengths)}")
        print(f"    Std: {np.std(lengths):.1f}")
        
        # Zero-length sequences (critical check!)
        zero_len = np.sum(lengths == 0)
        short_len = np.sum(lengths < 10)
        print(f"    Zero-length: {zero_len} {'[ERROR!]' if zero_len > 0 else '[OK]'}")
        print(f"    Short (<10 tokens): {short_len}")
        
        # Label distribution
        print(f"  Labels:")
        print(f"    Vulnerable (1): {np.sum(labels == 1)} ({100*np.mean(labels):.1f}%)")
        print(f"    Safe (0): {np.sum(labels == 0)} ({100*(1-np.mean(labels)):.1f}%)")
        
        # Token usage analysis (only for train)
        if split == 'train':
            print(f"\n  Token Usage Analysis:")
            
            # Count all token occurrences
            token_counts = Counter()
            unk_positions = []
            sep_positions = []
            
            unk_id = vocab.get('UNK', 1)
            pad_id = vocab.get('PAD', 0)
            sep_id = vocab.get('SEP', 4)
            
            for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
                actual_len = int(mask.sum())
                actual_ids = ids[:actual_len]
                
                for pos, tok_id in enumerate(actual_ids):
                    tok = id_to_token.get(tok_id, f'UNK_{tok_id}')
                    token_counts[tok] += 1
                    
                    if tok_id == unk_id:
                        unk_positions.append((i, pos))
                    if tok_id == sep_id:
                        sep_positions.append((i, pos))
            
            total_tokens = sum(token_counts.values())
            
            # UNK analysis
            unk_count = token_counts.get('UNK', 0)
            unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0
            print(f"    Total tokens: {total_tokens:,}")
            print(f"    UNK count: {unk_count} ({100*unk_ratio:.2f}%)")
            print(f"    UNK status: {'[OK] (< 1%)' if unk_ratio < 0.01 else '[Warning]'}")
            
            # SEP analysis
            sep_count = token_counts.get('SEP', 0)
            samples_with_sep = len(set(pos[0] for pos in sep_positions))
            sep_coverage = samples_with_sep / len(input_ids) if len(input_ids) > 0 else 0
            print(f"    SEP count: {sep_count}")
            print(f"    Samples with SEP: {samples_with_sep}/{len(input_ids)} ({100*sep_coverage:.1f}%)")
            print(f"    SEP status: {'[OK] (>99%)' if sep_coverage > 0.99 else '[Check multi-slicer]'}")
            
            # Top tokens
            print(f"\n    Top 20 tokens:")
            for tok, count in token_counts.most_common(20):
                print(f"      {tok}: {count} ({100*count/total_tokens:.2f}%)")
    
    return


def analyze_sep_insertion(data, vocab):
    """Detailed analysis of SEP token insertion."""
    print("\n" + "=" * 70)
    print("SEP TOKEN INSERTION ANALYSIS")
    print("=" * 70)
    
    id_to_token = {v: k for k, v in vocab.items()}
    sep_id = vocab.get('SEP', 4)
    
    for split in ['train']:  # Focus on train
        input_ids = data[split]['input_ids']
        attention_mask = data[split]['attention_mask']
        
        samples_with_sep = 0
        samples_without_sep = []
        sep_position_stats = []
        
        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            actual_len = int(mask.sum())
            actual_ids = ids[:actual_len]
            
            sep_positions = np.where(actual_ids == sep_id)[0]
            
            if len(sep_positions) > 0:
                samples_with_sep += 1
                # Record relative position of first SEP
                rel_pos = sep_positions[0] / actual_len if actual_len > 0 else 0
                sep_position_stats.append(rel_pos)
            else:
                samples_without_sep.append(i)
        
        total = len(input_ids)
        coverage = samples_with_sep / total if total > 0 else 0
        
        print(f"\n[{split.upper()}] SEP Coverage:")
        print(f"  With SEP: {samples_with_sep}/{total} ({100*coverage:.2f}%)")
        print(f"  Without SEP: {len(samples_without_sep)}")
        
        if sep_position_stats:
            print(f"\n  SEP Position (relative):")
            print(f"    Mean: {np.mean(sep_position_stats):.2f}")
            print(f"    Median: {np.median(sep_position_stats):.2f}")
            print(f"    Std: {np.std(sep_position_stats):.2f}")
        
        # Show examples without SEP (if any)
        if samples_without_sep and len(samples_without_sep) <= 20:
            print(f"\n  Samples without SEP (indices): {samples_without_sep[:20]}")
            
            # Decode first few
            print(f"\n  Sample sequences without SEP:")
            for idx in samples_without_sep[:3]:
                actual_len = int(attention_mask[idx].sum())
                tokens = [id_to_token.get(int(t), 'UNK') for t in input_ids[idx][:min(30, actual_len)]]
                print(f"    [{idx}] (len={actual_len}): {' '.join(tokens)}...")


def analyze_semantic_buckets(data, vocab):
    """Analyze semantic bucket usage."""
    print("\n" + "=" * 70)
    print("SEMANTIC BUCKET ANALYSIS")
    print("=" * 70)
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    bucket_prefixes = ['BUF_', 'LEN_', 'PTR_', 'IDX_', 'SENS_', 'PRIV_', 'VAR_']
    
    for split in ['train']:
        input_ids = data[split]['input_ids']
        attention_mask = data[split]['attention_mask']
        
        bucket_counts = {p.rstrip('_'): Counter() for p in bucket_prefixes}
        bucket_samples = {p.rstrip('_'): 0 for p in bucket_prefixes}
        
        for ids, mask in zip(input_ids, attention_mask):
            actual_len = int(mask.sum())
            actual_ids = ids[:actual_len]
            
            sample_buckets = set()
            
            for tok_id in actual_ids:
                tok = id_to_token.get(int(tok_id), 'UNK')
                for prefix in bucket_prefixes:
                    if tok.startswith(prefix):
                        bucket_name = prefix.rstrip('_')
                        bucket_counts[bucket_name][tok] += 1
                        sample_buckets.add(bucket_name)
            
            for bucket in sample_buckets:
                bucket_samples[bucket] += 1
        
        total_samples = len(input_ids)
        
        print(f"\n[{split.upper()}] Bucket Usage:")
        for bucket in ['BUF', 'LEN', 'PTR', 'IDX', 'SENS', 'PRIV', 'VAR']:
            sample_pct = 100 * bucket_samples[bucket] / total_samples if total_samples > 0 else 0
            unique_tokens = len(bucket_counts[bucket])
            total_uses = sum(bucket_counts[bucket].values())
            
            print(f"\n  {bucket}:")
            print(f"    Samples using: {bucket_samples[bucket]} ({sample_pct:.1f}%)")
            print(f"    Unique tokens: {unique_tokens}")
            print(f"    Total uses: {total_uses}")
            
            # Top tokens in bucket
            if bucket_counts[bucket]:
                top5 = bucket_counts[bucket].most_common(5)
                print(f"    Top 5: {dict(top5)}")


def analyze_api_preservation(data, vocab):
    """Analyze dangerous API preservation."""
    print("\n" + "=" * 70)
    print("API PRESERVATION ANALYSIS")
    print("=" * 70)
    
    from src.tokenization.hybrid_tokenizer import DANGEROUS_APIS
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Get API token IDs
    api_ids = {vocab[api]: api for api in DANGEROUS_APIS if api in vocab}
    
    for split in ['train']:
        input_ids = data[split]['input_ids']
        attention_mask = data[split]['attention_mask']
        
        api_counts = Counter()
        samples_with_api = 0
        
        for ids, mask in zip(input_ids, attention_mask):
            actual_len = int(mask.sum())
            actual_ids = ids[:actual_len]
            
            has_api = False
            for tok_id in actual_ids:
                if int(tok_id) in api_ids:
                    api_counts[api_ids[int(tok_id)]] += 1
                    has_api = True
            
            if has_api:
                samples_with_api += 1
        
        total_samples = len(input_ids)
        api_coverage = samples_with_api / total_samples if total_samples > 0 else 0
        
        print(f"\n[{split.upper()}] API Preservation:")
        print(f"  APIs in vocab: {len(api_ids)}")
        print(f"  Samples with APIs: {samples_with_api}/{total_samples} ({100*api_coverage:.1f}%)")
        
        print(f"\n  API Frequency (top 20):")
        for api, count in api_counts.most_common(20):
            print(f"    {api}: {count}")
        
        # Categories of APIs
        memory_apis = ['malloc', 'calloc', 'realloc', 'free', 'alloca']
        string_apis = ['strcpy', 'strcat', 'strncpy', 'strncat', 'sprintf', 'snprintf']
        io_apis = ['printf', 'fprintf', 'scanf', 'fscanf', 'fread', 'fwrite', 'read', 'write']
        mem_ops = ['memcpy', 'memmove', 'memset', 'memcmp']
        
        print(f"\n  API Categories:")
        for name, apis in [('Memory Alloc', memory_apis), ('String', string_apis), 
                           ('I/O', io_apis), ('Memory Ops', mem_ops)]:
            total = sum(api_counts.get(api, 0) for api in apis)
            present = [api for api in apis if api_counts.get(api, 0) > 0]
            print(f"    {name}: {total} uses, present: {present}")


def analyze_features(data):
    """Analyze vulnerability features."""
    print("\n" + "=" * 70)
    print("VULNERABILITY FEATURES ANALYSIS")
    print("=" * 70)
    
    for split in ['train']:
        features = data[split]['features']
        feature_names = list(data[split]['feature_names'])
        
        print(f"\n[{split.upper()}] Features:")
        print(f"  Shape: {features.shape}")
        print(f"  Feature count: {len(feature_names)}")
        
        # Non-zero analysis
        non_zero_per_feature = np.count_nonzero(features, axis=0)
        non_zero_per_sample = np.count_nonzero(features, axis=1)
        
        print(f"\n  Non-zero features per sample:")
        print(f"    Mean: {np.mean(non_zero_per_sample):.1f}")
        print(f"    Min: {np.min(non_zero_per_sample)}, Max: {np.max(non_zero_per_sample)}")
        
        print(f"\n  Feature statistics:")
        for i, name in enumerate(feature_names):
            col = features[:, i]
            non_zero = non_zero_per_feature[i]
            mean = np.mean(col)
            std = np.std(col)
            max_val = np.max(col)
            
            if non_zero > 0 or 'count' in name:
                print(f"    {name}: non-zero={non_zero}, mean={mean:.3f}, std={std:.3f}, max={max_val:.1f}")


def decode_random_samples(data, vocab, n=5):
    """Decode and display random samples."""
    print("\n" + "=" * 70)
    print("RANDOM SAMPLE DECODING")
    print("=" * 70)
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    for split in ['train']:
        input_ids = data[split]['input_ids']
        attention_mask = data[split]['attention_mask']
        labels = data[split]['labels']
        
        # Random indices
        np.random.seed(42)
        indices = np.random.choice(len(input_ids), min(n, len(input_ids)), replace=False)
        
        print(f"\n[{split.upper()}] Random Samples:")
        
        for idx in indices:
            actual_len = int(attention_mask[idx].sum())
            tokens = [id_to_token.get(int(t), f'UNK_{t}') for t in input_ids[idx][:actual_len]]
            label = 'VULNERABLE' if labels[idx] == 1 else 'SAFE'
            
            print(f"\n  Sample {idx} (label={label}, len={actual_len}):")
            
            # Find SEP position
            sep_pos = None
            for i, t in enumerate(tokens):
                if t == 'SEP':
                    sep_pos = i
                    break
            
            if sep_pos:
                print(f"    [BACKWARD] {' '.join(tokens[:sep_pos][:50])}...")
                print(f"    [SEP]")
                print(f"    [FORWARD] {' '.join(tokens[sep_pos+1:][:50])}...")
            else:
                print(f"    {' '.join(tokens[:100])}...")


def main():
    print("=" * 70)
    print("DEVIGN PREPROCESSING V2 - COMPREHENSIVE DEBUG ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    vocab, config, data = load_data()
    
    # Run analyses
    analyze_vocab(vocab)
    analyze_sequences(data, vocab)
    analyze_sep_insertion(data, vocab)
    analyze_semantic_buckets(data, vocab)
    analyze_api_preservation(data, vocab)
    analyze_features(data)
    decode_random_samples(data, vocab, n=5)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    train_data = data['train']
    lengths = train_data['attention_mask'].sum(axis=1)
    zero_len = np.sum(lengths == 0)
    
    id_to_token = {v: k for k, v in vocab.items()}
    sep_id = vocab.get('SEP', 4)
    samples_with_sep = 0
    for ids, mask in zip(train_data['input_ids'], train_data['attention_mask']):
        if sep_id in ids[:int(mask.sum())]:
            samples_with_sep += 1
    
    sep_coverage = samples_with_sep / len(train_data['input_ids'])
    
    print(f"\n  [OK] Vocab size: {len(vocab)}")
    print(f"  {'[OK]' if zero_len == 0 else '[ERROR]'} Zero-length samples: {zero_len}")
    print(f"  {'[OK]' if sep_coverage > 0.99 else '[WARN]'} SEP coverage: {100*sep_coverage:.1f}%")
    print(f"  [OK] Train samples: {len(train_data['input_ids'])}")
    print(f"  [OK] Feature count: {train_data['features'].shape[1]}")
    
    print("\n" + "=" * 70)
    print("DEBUG ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

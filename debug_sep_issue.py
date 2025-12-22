"""
Debug SEP token insertion issue.
According to Oracle analysis, SEP is only 68.2% because:
1. _combine_slices returns single slice when one is empty (no SEP)
2. Fallback replaces combined with full function (no SEP)
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path('F:/Work/C Vul Devign/Dataset/devign_slice_v2')

def main():
    print("=" * 70)
    print("SEP TOKEN ISSUE INVESTIGATION")
    print("=" * 70)
    
    # Load vocab
    with open(OUTPUT_DIR / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    
    id_to_token = {v: k for k, v in vocab.items()}
    sep_id = vocab.get('SEP', 4)
    
    # Load train data
    data = np.load(OUTPUT_DIR / 'train.npz')
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']
    
    print(f"\nTotal samples: {len(input_ids)}")
    
    # Analyze SEP presence
    samples_with_sep = []
    samples_without_sep = []
    
    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        actual_len = int(mask.sum())
        actual_ids = ids[:actual_len]
        
        if sep_id in actual_ids:
            samples_with_sep.append(i)
        else:
            samples_without_sep.append(i)
    
    print(f"\nWith SEP: {len(samples_with_sep)} ({100*len(samples_with_sep)/len(input_ids):.1f}%)")
    print(f"Without SEP: {len(samples_without_sep)} ({100*len(samples_without_sep)/len(input_ids):.1f}%)")
    
    # Analyze samples WITHOUT SEP
    print("\n" + "=" * 70)
    print("ANALYZING SAMPLES WITHOUT SEP")
    print("=" * 70)
    
    # Length distribution
    no_sep_lengths = [int(attention_mask[i].sum()) for i in samples_without_sep]
    with_sep_lengths = [int(attention_mask[i].sum()) for i in samples_with_sep]
    
    print(f"\nToken length comparison:")
    print(f"  WITHOUT SEP: mean={np.mean(no_sep_lengths):.1f}, median={np.median(no_sep_lengths):.1f}")
    print(f"  WITH SEP: mean={np.mean(with_sep_lengths):.1f}, median={np.median(with_sep_lengths):.1f}")
    
    # Label distribution
    no_sep_labels = [labels[i] for i in samples_without_sep]
    with_sep_labels = [labels[i] for i in samples_with_sep]
    
    print(f"\nLabel distribution:")
    print(f"  WITHOUT SEP: vuln={sum(no_sep_labels)} ({100*np.mean(no_sep_labels):.1f}%)")
    print(f"  WITH SEP: vuln={sum(with_sep_labels)} ({100*np.mean(with_sep_labels):.1f}%)")
    
    # Check for patterns in no-SEP samples
    print("\n" + "=" * 70)
    print("SAMPLE TOKENS (WITHOUT SEP)")
    print("=" * 70)
    
    # Decode first 10 samples without SEP
    for idx in samples_without_sep[:10]:
        actual_len = int(attention_mask[idx].sum())
        tokens = [id_to_token.get(int(t), f'UNK_{t}') for t in input_ids[idx][:min(60, actual_len)]]
        label = 'VULN' if labels[idx] == 1 else 'SAFE'
        
        print(f"\n[{idx}] label={label}, len={actual_len}")
        print(f"  {' '.join(tokens)}...")
    
    # Check if they have special patterns
    print("\n" + "=" * 70)
    print("TOKEN DISTRIBUTION (WITHOUT SEP)")
    print("=" * 70)
    
    no_sep_token_counts = Counter()
    for idx in samples_without_sep:
        actual_len = int(attention_mask[idx].sum())
        actual_ids = input_ids[idx][:actual_len]
        for tok_id in actual_ids:
            tok = id_to_token.get(int(tok_id), f'UNK_{tok_id}')
            no_sep_token_counts[tok] += 1
    
    print("\nTop 20 tokens in no-SEP samples:")
    for tok, count in no_sep_token_counts.most_common(20):
        print(f"  {tok}: {count}")
    
    # Check if there's a pattern of "short" samples without SEP
    print("\n" + "=" * 70)
    print("LENGTH DISTRIBUTION (WITHOUT SEP)")
    print("=" * 70)
    
    length_bins = {
        '< 10': sum(1 for l in no_sep_lengths if l < 10),
        '10-50': sum(1 for l in no_sep_lengths if 10 <= l < 50),
        '50-100': sum(1 for l in no_sep_lengths if 50 <= l < 100),
        '100-200': sum(1 for l in no_sep_lengths if 100 <= l < 200),
        '200-512': sum(1 for l in no_sep_lengths if 200 <= l <= 512),
    }
    
    print("\nLength distribution of no-SEP samples:")
    for bin_name, count in length_bins.items():
        print(f"  {bin_name}: {count} ({100*count/len(samples_without_sep):.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print("""
Based on Oracle analysis, the 6942 samples without SEP are caused by:

1. EMPTY SLICE CASE:
   In multi_slicer.py _combine_slices():
   - If backward_code is empty -> returns forward_code only (NO SEP)
   - If forward_code is empty -> returns backward_code only (NO SEP)
   
2. FALLBACK CASE:
   In 03_preprocessing_v2.py process_multi_slice_batch():
   - If combined_code has < MIN_SLICE_TOKENS (10) -> replaces with full function code
   - Full function code does NOT have [SEP] inserted
   
SOLUTION:
1. Modify _combine_slices to ALWAYS insert [SEP] even if one slice is empty
2. Modify fallback to insert [SEP] in the middle of full function

For BiGRU model, SEP token helps separate backward (context) from forward (effect)
slices, which is important for learning vulnerability patterns.
""")


if __name__ == '__main__':
    main()

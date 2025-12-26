import numpy as np

train = np.load('Dataset/devign_final/train.npz')
val = np.load('Dataset/devign_final/val.npz')
test = np.load('Dataset/devign_final/test.npz')

print("=" * 60)
print("CLASS DISTRIBUTION ANALYSIS")
print("=" * 60)

for name, data in [('Train', train), ('Val', val), ('Test', test)]:
    labels = data['labels']
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = len(labels)
    print(f'{name}: total={total}, pos={n_pos} ({100*n_pos/total:.1f}%), neg={n_neg} ({100*n_neg/total:.1f}%)')

# Check UNK and PAD tokens
input_ids = train['input_ids']
print(f'\n{"=" * 60}')
print("TOKEN ANALYSIS (Train set)")
print("=" * 60)
print(f'Input shape: {input_ids.shape}')
unk_count = (input_ids == 1).sum()
pad_count = (input_ids == 0).sum()
total_tokens = input_ids.size
non_pad = (input_ids > 0).sum()

print(f'PAD (id=0): {pad_count} ({100*pad_count/total_tokens:.1f}%)')
print(f'UNK (id=1): {unk_count} ({100*unk_count/non_pad:.2f}% of non-PAD)')
print(f'Non-PAD tokens: {non_pad}')

# Token diversity per sample
unique_per_sample = []
for i in range(min(1000, len(input_ids))):
    seq = input_ids[i]
    non_pad_tokens = seq[seq > 0]
    unique_per_sample.append(len(set(non_pad_tokens)))

print(f'\nUnique tokens per sample (first 1000):')
print(f'  Mean: {np.mean(unique_per_sample):.1f}')
print(f'  Std: {np.std(unique_per_sample):.1f}')
print(f'  Min: {np.min(unique_per_sample)}')
print(f'  Max: {np.max(unique_per_sample)}')

# Check for duplicate sequences
from collections import Counter
seq_hashes = [hash(tuple(seq.tolist())) for seq in input_ids[:5000]]
hash_counts = Counter(seq_hashes)
n_duplicates = sum(1 for h, c in hash_counts.items() if c > 1)
print(f'\nDuplicate sequences (in first 5000): {n_duplicates} unique hashes with duplicates')

# Sequence length distribution
seq_lengths = (input_ids > 0).sum(axis=1)
print(f'\nSequence lengths:')
print(f'  Mean: {np.mean(seq_lengths):.1f}')
print(f'  Median: {np.median(seq_lengths):.1f}')
print(f'  Min: {np.min(seq_lengths)}')
print(f'  Max: {np.max(seq_lengths)}')

# Token frequency analysis
all_tokens = input_ids[input_ids > 0].flatten()
token_counts = Counter(all_tokens)
print(f'\nMost common tokens (top 20):')
for token_id, count in token_counts.most_common(20):
    print(f'  Token {token_id}: {count}')

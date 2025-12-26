import numpy as np
import json
from collections import Counter

train = np.load('Dataset/devign_final/train.npz')
X = train['input_ids']
y = train['labels']
vocab = json.load(open('Dataset/devign_final/vocab.json'))
id2token = {v:k for k,v in vocab.items()}

pos_counts = Counter()
neg_counts = Counter()
for seq, label in zip(X, y):
    tokens = [id2token.get(tid,'UNK') for tid in seq if tid > 0]
    if label == 1:
        pos_counts.update(tokens)
    else:
        neg_counts.update(tokens)

# Find most discriminative tokens
all_tokens = set(pos_counts.keys()) | set(neg_counts.keys())
diffs = []
for t in all_tokens:
    pc = pos_counts.get(t, 0)
    nc = neg_counts.get(t, 0)
    total = pc + nc
    if total >= 100:  # Minimum frequency
        ratio = pc / total
        diffs.append((t, pc, nc, ratio, abs(ratio - 0.5)))

# Sort by discriminative power
diffs.sort(key=lambda x: -x[4])
print('Most discriminative tokens (>100 occurrences):')
print(f'{"Token":20} | {"Pos":>6} | {"Neg":>6} | Ratio | Diff')
for t, pc, nc, ratio, diff in diffs[:25]:
    marker = "VUL" if ratio > 0.5 else "SAFE"
    print(f'{t:20} | {pc:6} | {nc:6} | {ratio:.3f} | {diff:.3f} {marker}')

# Check if tokens are too uniformly distributed
print('\n=== Token distribution summary ===')
low_signal = sum(1 for _, _, _, _, d in diffs if d < 0.05)
med_signal = sum(1 for _, _, _, _, d in diffs if 0.05 <= d < 0.1)
high_signal = sum(1 for _, _, _, _, d in diffs if d >= 0.1)
print(f'Low signal (diff<0.05): {low_signal} tokens')
print(f'Medium signal (0.05-0.1): {med_signal} tokens')
print(f'High signal (diff>=0.1): {high_signal} tokens')

import numpy as np
import json

# Load data
train = np.load('Dataset/devign_slice_v2/train.npz')
X = train['input_ids']
y = train['labels']
attn = train['attention_mask']

with open('Dataset/devign_slice_v2/vocab.json') as f:
    vocab = json.load(f)

id2tok = {v: k for k, v in vocab.items()}

print('=' * 60)
print('DATA INFO')
print('=' * 60)
print(f'X shape: {X.shape}')
print(f'Vocab size: {len(vocab)}')

# Decode sample 0
print('\n' + '=' * 60)
print(f'SAMPLE 0 (label={y[0]})')
print('=' * 60)
tokens0 = [id2tok.get(int(i), f'?{i}?') for i in X[0] if i != 0]
print(f'Non-pad length: {len(tokens0)}')
print(f'Decoded tokens:\n{" ".join(tokens0)}')

# Decode sample 1
print('\n' + '=' * 60)
print(f'SAMPLE 1 (label={y[1]})')
print('=' * 60)
tokens1 = [id2tok.get(int(i), f'?{i}?') for i in X[1] if i != 0]
print(f'Non-pad length: {len(tokens1)}')
print(f'Decoded tokens:\n{" ".join(tokens1)}')

# Stats
print('\n' + '=' * 60)
print('TOKEN STATS')
print('=' * 60)
pad_count = np.sum(X == 0)
unk_count = np.sum(X == 1)
total = X.size
print(f'PAD (0): {pad_count:,} ({pad_count/total*100:.2f}%)')
print(f'UNK (1): {unk_count:,} ({unk_count/total*100:.2f}%)')

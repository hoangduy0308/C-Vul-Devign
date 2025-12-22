#!/usr/bin/env python3
"""Analyze devign_slice_v2 output"""
import numpy as np
import json

# Load vocab
with open('Dataset/devign_slice_v2/vocab.json') as f:
    vocab = json.load(f)

print('=' * 60)
print('VOCAB ANALYSIS')
print('=' * 60)
print(f'Vocab size: {len(vocab)}')

# Create reverse vocab
id_to_token = {v: k for k, v in vocab.items()}

# Load train data
train = np.load('Dataset/devign_slice_v2/train.npz')
print('\n' + '=' * 60)
print('TRAIN DATA SHAPE')
print('=' * 60)
print(f'Keys: {list(train.keys())}')
print(f'X shape: {train["X"].shape}')
print(f'y shape: {train["y"].shape}')

X = train['X']
y = train['y']

# Token distribution
print('\n' + '=' * 60)
print('TOKEN DISTRIBUTION')
print('=' * 60)
unique, counts = np.unique(X, return_counts=True)
print(f'Unique token IDs used: {len(unique)}')
print(f'\nTop 20 tokens by frequency:')
sorted_idx = np.argsort(-counts)[:20]
for idx in sorted_idx:
    token_id = unique[idx]
    token_name = id_to_token.get(token_id, f'UNKNOWN_ID_{token_id}')
    print(f'  ID {token_id:3d} ({token_name:>10s}): {counts[idx]:,}')

# Padding and UNK analysis
print('\n' + '=' * 60)
print('PADDING / UNK ANALYSIS')
print('=' * 60)
total = X.size
pad_count = np.sum(X == 0)
unk_count = np.sum(X == 1)
print(f'Total tokens: {total:,}')
print(f'PAD (id=0): {pad_count:,} ({pad_count/total*100:.2f}%)')
print(f'UNK (id=1): {unk_count:,} ({unk_count/total*100:.2f}%)')

# Sequence length check
print('\n' + '=' * 60)
print('SEQUENCE LENGTH VERIFICATION')
print('=' * 60)
print(f'Expected max_len: 512')
print(f'Actual sequence length: {X.shape[1]}')

# Sample sequences decoded
print('\n' + '=' * 60)
print('SAMPLE DECODED SEQUENCES')
print('=' * 60)

def decode_sequence(ids, id_to_token, max_display=50):
    tokens = [id_to_token.get(id, f'?{id}?') for id in ids if id != 0]  # Skip PAD
    return tokens[:max_display]

for i in range(3):
    tokens = decode_sequence(X[i], id_to_token, 40)
    print(f'\nSample {i} (label={y[i]}):')
    print(f'  First 40 non-pad tokens: {tokens}')
    
    # Count non-pad tokens
    non_pad = np.sum(X[i] != 0)
    print(f'  Non-pad tokens: {non_pad}')

# Check for any tokens > vocab_size
print('\n' + '=' * 60)
print('TOKEN ID RANGE CHECK')
print('=' * 60)
max_id = np.max(X)
min_id = np.min(X)
print(f'Token ID range: {min_id} to {max_id}')
print(f'Vocab size: {len(vocab)}')
if max_id >= len(vocab):
    print(f'WARNING: Found token IDs >= vocab size!')
    invalid = X[X >= len(vocab)]
    print(f'Invalid token IDs: {np.unique(invalid)}')
else:
    print('OK: All token IDs are valid')

# Vuln features check
print('\n' + '=' * 60)
print('VULNERABILITY FEATURES')
print('=' * 60)
train_vuln = np.load('Dataset/devign_slice_v2/train_vuln.npz')
print(f'Keys: {list(train_vuln.keys())}')
print(f'X_vuln shape: {train_vuln["X_vuln"].shape}')

X_vuln = train_vuln['X_vuln']
print(f'\nFeature stats (first 5 features):')
for i in range(min(5, X_vuln.shape[1])):
    print(f'  Feature {i}: min={X_vuln[:,i].min():.3f}, max={X_vuln[:,i].max():.3f}, mean={X_vuln[:,i].mean():.3f}')

print('\n' + '=' * 60)
print('ANALYSIS COMPLETE')
print('=' * 60)

import numpy as np
import json

train = np.load('Dataset/devign_slice_v2/train.npz')
X = train['input_ids']
y = train['labels']

with open('Dataset/devign_slice_v2/vocab.json') as f:
    vocab = json.load(f)

id2tok = {v: k for k, v in vocab.items()}

# Count dangerous API occurrences
dangerous_apis = ['malloc', 'free', 'strcpy', 'strncpy', 'memcpy', 'memset', 
                  'sprintf', 'snprintf', 'gets', 'fgets', 'scanf', 'printf',
                  'read', 'write', 'open', 'close', 'fopen', 'fclose']

print('=== DANGEROUS API TOKEN IDS ===')
for api in dangerous_apis:
    if api in vocab:
        print(f'{api}: id={vocab[api]}')

print('\n=== DANGEROUS API COUNTS IN TRAIN DATA ===')
total_dangerous = 0
for api in dangerous_apis:
    if api in vocab:
        api_id = vocab[api]
        count = np.sum(X == api_id)
        total_dangerous += count
        print(f'{api} (id={api_id}): {count:,} occurrences')

print(f'\nTotal dangerous API tokens: {total_dangerous:,}')

# Count FUNC and ID
func_id = vocab.get('FUNC', -1)
id_id = vocab.get('ID', -1)
print(f'\nFUNC tokens: {np.sum(X == func_id):,}')
print(f'ID tokens: {np.sum(X == id_id):,}')

# Find samples with malloc
print('\n=== SAMPLE WITH malloc ===')
malloc_id = vocab.get('malloc', -1)
if malloc_id >= 0:
    samples_with_malloc = np.where(np.any(X == malloc_id, axis=1))[0]
    print(f'Found {len(samples_with_malloc)} samples with malloc')
    if len(samples_with_malloc) > 0:
        idx = samples_with_malloc[0]
        tokens = [id2tok.get(int(i), '?') for i in X[idx] if i != 0]
        print(f'\nSample {idx} (label={y[idx]}):')
        print(' '.join(tokens[:120]))

# Find samples with strcpy
print('\n=== SAMPLE WITH strcpy ===')
strcpy_id = vocab.get('strcpy', -1)
if strcpy_id >= 0:
    samples_with_strcpy = np.where(np.any(X == strcpy_id, axis=1))[0]
    print(f'Found {len(samples_with_strcpy)} samples with strcpy')
    if len(samples_with_strcpy) > 0:
        idx = samples_with_strcpy[0]
        tokens = [id2tok.get(int(i), '?') for i in X[idx] if i != 0]
        print(f'\nSample {idx} (label={y[idx]}):')
        print(' '.join(tokens[:120]))

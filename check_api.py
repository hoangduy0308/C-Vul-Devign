import numpy as np
import json

data = np.load('Dataset/devign_final/train.npz')
tokens = data['input_ids']
labels = data['labels']
vocab = json.load(open('Dataset/devign_final/vocab.json', encoding='utf-8'))

funcs = [('free', 18), ('malloc', 124), ('memcpy', 161), ('strcpy', 69), 
         ('av_freep', 29), ('g_free', 44), ('realloc', 232), ('sprintf', vocab.get('sprintf', -1))]

print("Dangerous API occurrence analysis:")
print("="*50)

for name, fid in funcs:
    if fid == -1:
        print(f"{name}: NOT IN VOCAB")
        continue
    vuln = 0
    safe = 0
    for i, seq in enumerate(tokens):
        for t in seq:
            if int(t) == fid:
                if labels[i] == 1:
                    vuln += 1
                else:
                    safe += 1
    ratio = vuln/safe if safe > 0 else float('inf')
    print(f"{name:12} vuln={vuln:5}  safe={safe:5}  ratio={ratio:.2f}")

# Check total dangerous api tokens in vuln vs safe
dangerous_ids = set([18, 124, 161, 69, 29, 44, 232])
vuln_total = 0
safe_total = 0
for i, seq in enumerate(tokens):
    count = sum(1 for t in seq if int(t) in dangerous_ids)
    if labels[i] == 1:
        vuln_total += count
    else:
        safe_total += count

print(f"\nTotal dangerous API tokens: vuln={vuln_total}, safe={safe_total}")
print(f"Ratio: {vuln_total/safe_total:.2f}" if safe_total > 0 else "N/A")

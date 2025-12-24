import json
import numpy as np

all_lengths = []
vul_lengths = []
non_vul_lengths = []

for split in ['train', 'val', 'test']:
    filepath = f'F:/Work/C Vul Devign/Dataset/devign_final/{split}_tokens.jsonl'
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokens = data['tokens']
            label = data.get('label', 0)
            length = len(tokens)
            all_lengths.append(length)
            if label == 1:
                vul_lengths.append(length)
            else:
                non_vul_lengths.append(length)

all_lengths = np.array(all_lengths)
vul_lengths = np.array(vul_lengths)
non_vul_lengths = np.array(non_vul_lengths)

percentiles = [50, 75, 90, 95, 97, 99]

print('='*60)
print('SEQUENCE LENGTH ANALYSIS - Devign Dataset')
print('='*60)
print(f'\nTotal samples: {len(all_lengths):,}')
print(f'  - Vulnerable: {len(vul_lengths):,}')
print(f'  - Non-vulnerable: {len(non_vul_lengths):,}')

print('\n' + '-'*60)
print('OVERALL PERCENTILES')
print('-'*60)
print(f'{"Percentile":<12} {"Length":>10}')
print('-'*24)
for p in percentiles:
    val = np.percentile(all_lengths, p)
    print(f'P{p:<11} {val:>10.0f}')
print(f'{"Max":<12} {np.max(all_lengths):>10}')
print(f'{"Min":<12} {np.min(all_lengths):>10}')
print(f'{"Mean":<12} {np.mean(all_lengths):>10.1f}')

print('\n' + '-'*60)
print('VULNERABLE vs NON-VULNERABLE')
print('-'*60)
print(f'{"Percentile":<12} {"Vulnerable":>12} {"Non-Vul":>12}')
print('-'*36)
for p in percentiles:
    vul_val = np.percentile(vul_lengths, p) if len(vul_lengths) > 0 else 0
    non_vul_val = np.percentile(non_vul_lengths, p) if len(non_vul_lengths) > 0 else 0
    print(f'P{p:<11} {vul_val:>12.0f} {non_vul_val:>12.0f}')
print(f'{"Max":<12} {np.max(vul_lengths):>12} {np.max(non_vul_lengths):>12}')

print('\n' + '-'*60)
print('COVERAGE ANALYSIS (% samples covered)')
print('-'*60)
for max_len in [256, 384, 512, 768, 1024]:
    coverage = np.mean(all_lengths <= max_len) * 100
    vul_cov = np.mean(vul_lengths <= max_len) * 100 if len(vul_lengths) > 0 else 0
    non_vul_cov = np.mean(non_vul_lengths <= max_len) * 100 if len(non_vul_lengths) > 0 else 0
    print(f'max_len={max_len:<4}: {coverage:6.2f}% overall | Vul: {vul_cov:6.2f}% | Non-vul: {non_vul_cov:6.2f}%')

p95 = np.percentile(all_lengths, 95)
print('\n' + '='*60)
print('RECOMMENDATION')
print('='*60)
if p95 <= 256:
    rec = 256
elif p95 <= 384:
    rec = 384
elif p95 <= 512:
    rec = 512
else:
    rec = 768
print(f'P95 = {p95:.0f} tokens')
print(f'Recommended max_length: {rec}')
print(f'Coverage at {rec}: {np.mean(all_lengths <= rec)*100:.2f}%')

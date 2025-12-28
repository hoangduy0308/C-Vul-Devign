import json
from collections import defaultdict

data = []
with open('F:/Work/C Vul Devign/Dataset/devign_final/debug/train.slices_tokens.jsonl','r',encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        data.append((d['sliced_code'], d['label'], d['sample_id'], d['tokens']))

print(f"Total samples: {len(data)}")

# Group by code
by_code = defaultdict(list)
for code, label, sid, tokens in data:
    by_code[code].append((label, sid, tokens))

print(f"Unique codes: {len(by_code)}")

# Find conflicts
conflicts = [(code, items) for code, items in by_code.items() if len(set(l for l,s,t in items)) > 1]
print(f"Conflicts (same code, diff labels): {len(conflicts)}")

# Show first 2 in detail
for i, (code, items) in enumerate(conflicts[:2]):
    print(f"\n=== CONFLICT {i+1} ===")
    print(f"Code (exact):")
    print(repr(code[:300]))
    print()
    for label, sid, tokens in items:
        print(f"  Sample {sid}: label={label}, n_tokens={len(tokens)}")

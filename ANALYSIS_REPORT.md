# Analysis Report - Root Cause & Fixes for Low AUC

## Executive Summary

After comprehensive analysis using Oracle, GitHub research, and dataset inspection, the **root cause** of the model predicting 94%+ samples as vulnerable (AUC=0.587) has been identified:

### 🔥 Root Cause: Token Collapse

**The tokenizer configuration `use_indexed_buckets=False` destroys discriminative information.**

| Token Type | Expected (indexed) | Actual (collapsed) | Impact |
|------------|-------------------|-------------------|--------|
| Variables | VAR_0, VAR_1, VAR_2 | VAR | Cannot track same variable across slice |
| Buffers | BUF_0, BUF_1 | BUF | Cannot distinguish source vs dest |
| Lengths | LEN_0, LEN_1 | LEN | Cannot correlate len with buffer |
| Functions | FUNC | FUNC | OK (only 1 FUNC token) |

### Evidence from Dataset Analysis

```
Train: 21,808 samples (45.8% vuln / 54.2% safe) ← BALANCED!
Val: 2,724 samples (43.5% vuln / 56.5% safe)
Test: 2,726 samples (45.9% vuln / 54.1% safe)

Token Statistics (Train):
- Most common: VAR=467,942, BUF=180,775, FUNC=128,467
- Unique tokens per sample: only 27.7 average (from 410 vocab)
- UNK rate: 0.00% (good)
```

**The model sees sequences like:**
```
memcpy ( BUF , VAR , LEN ) ; if ( LEN < VAR ) ...
```

**But it SHOULD see:**
```
memcpy ( BUF_0 , VAR_0 , LEN_0 ) ; if ( LEN_0 < CAP_0 ) ...
```

Without indices, the model cannot learn:
- "Same buffer BUF_0 flows from source to destination"
- "LEN_0 is checked against CAP_0 (safe) vs LEN_0 is unchecked (vulnerable)"

---

## Action Plan

### Fix 1: Enable Indexed Buckets (CRITICAL)

File: `devign_pipeline/notebooks/preprocessing_full.py` line 139

```python
# BEFORE
'use_indexed_buckets': False,  # ❌ Destroys discriminative info

# AFTER
'use_indexed_buckets': True,   # ✅ Preserves VAR_0, BUF_0, LEN_0 distinct tokens
```

### Fix 2: Replace Focal Loss with BCEWithLogitsLoss (Sanity Check)

File: `devign_pipeline/src/training/config_simplified.py`

```python
# BEFORE (get_focal_config was used)
loss_type: str = "focal_alpha"
focal_alpha: float = 0.25

# AFTER (use balanced BCE first)
loss_type: str = "bce_weighted"
pos_weight_override: float = 1.0  # Data is balanced
```

### Fix 3: Change Threshold Optimization Metric

```python
# BEFORE
threshold_optimization_metric: str = 'f1'  # Biases toward all-positive on balanced data

# AFTER
threshold_optimization_metric: str = 'mcc'  # Or 'balanced' (Youden's J)
```

---

## Research Findings (GitHub/Papers)

| Implementation | Approach | AUC/F1 | Key Insight |
|---------------|----------|--------|-------------|
| [LineVul](https://github.com/awsm-research/LineVul) | CodeBERT + BPE | F1=0.91 | Pretrained embeddings crucial |
| [epicosy/devign](https://github.com/epicosy/devign) | GNN + AST | AUC=0.54-0.56 | Uses VAR1, VAR2, FUN1, FUN2 numbering |
| [Function-level-VD](https://github.com/DanielLin1986/Function-level-Vulnerability-Detection) | BiGRU + balanced sampling | F1~0.70 | `class_weight='balanced'` |

**Key takeaway**: All successful implementations preserve variable/function numbering (VAR1, VAR2, etc.) to track data flow!

---: Why Model Still Predicts All Vulnerable

## Summary of Findings

### Current Results (After Tokenizer Changes)
- **F1**: 0.622 (worse than baseline 0.649)
- **Precision**: 46.1% (should be ~50%+)  
- **Recall**: 95.8% (too high - predicting almost everything positive)
- **AUC**: 0.587 (barely better than random 0.5)
- **Optimal Threshold**: 0.24 (should be ~0.5)

### Root Cause: Token Abstraction Too Aggressive

The vocabulary reduction from 30k to 389 tokens **destroyed discriminative signals**:

1. **65% of tokens have |ratio - 0.5| < 0.05** → nearly zero discriminative power
2. **Indexed buckets don't have consistent semantics across samples**:
   - `BUF_0` in sample A might be destination, in sample B might be source
   - The model can't learn "BUF_0 = dangerous" because it means different things
3. **Most discriminative tokens are random indexed ones** (`SENS_3`, `PTR_4`, `CMD_4`)
   - These correlate with vuln/safe by chance, not semantically

### Token Distribution Evidence

```
Token           | Pos Count | Neg Count | Ratio | Signal
----------------|-----------|-----------|-------|--------
SENS_3          |        33 |        84 | 0.282 | SAFE (0.218)
PTR_4           |       159 |        79 | 0.668 | VUL (0.168)
DEF_BOUNDS      |       149 |       220 | 0.404 | SAFE (0.096)
DEF_CHECK       |       747 |      1033 | 0.419 | SAFE (0.081)
BUF_0           |     23429 |     27401 | 0.461 | (0.039) ← NO SIGNAL
LEN_0           |     12161 |     12214 | 0.499 | (0.001) ← NO SIGNAL  
NULL            |      8387 |      8644 | 0.492 | (0.008) ← NO SIGNAL
```

**Defense tokens (DEF_*) have weak signal but ARE working** - they appear more in safe code.
**Bucket tokens (BUF_0, LEN_0) have ZERO signal** - indexed approach failed.

---

## Recommended Solutions

### Solution 1: Disable Indexed Buckets (High Impact, Low Effort)
Set `use_indexed_buckets=False` → tokens become `BUF`, `LEN`, `PTR` instead of `BUF_0`, `BUF_1`...

**Why this helps:**
- Increases token frequency (better Word2Vec embeddings)
- Removes random aliasing between semantically different variables
- Forces model to learn "presence of buffer-like variable" rather than specific indices

### Solution 2: Expand Preserved Exact APIs (Medium Impact)
Add more security-critical APIs to TRULY_DANGEROUS_APIS and preserve safe versions:

```python
# Dangerous (keep exact)
TRULY_DANGEROUS_APIS = {
    'gets', 'strcpy', 'strcat', 'sprintf', 'vsprintf', 'scanf', 'sscanf',
    'system', 'popen', 'execve', 'memcpy', 'memmove',  # ADD THESE
}

# Safe versions (add new SAFE_APIS set)
SAFE_APIS = {
    'snprintf', 'strncpy', 'strlcpy', 'strncat', 'strlcat',
    'memcpy_s', 'strcpy_s', 'strncpy_s',
}
```

**Why this helps:**
- Distinguishes `strcpy` (dangerous) from `strncpy` (safer)
- Current API_COPY family merges them all

### Solution 3: Increase Vocabulary Size to 2000-5000
Current 389 is too aggressive. Target ~2000-5000 with smart selection:

1. Keep all C keywords and operators
2. Keep exact dangerous APIs (not families)
3. Keep exact safe APIs
4. Keep defense function names
5. Add top-1000 most frequent identifiers from training data

### Solution 4: Use Focal Loss Instead of BCE
Current F1-optimized thresholding + BCE creates "predict all positive" artifact.

```python
# In config
config.loss_type = 'focal_alpha'
config.focal_alpha = 0.25
config.focal_gamma = 2.0
```

### Solution 5: Remove F1 Threshold Optimization (Diagnostic)
For debugging, fix threshold at 0.5 to see true model behavior:

```python
# In evaluate()
preds = (probs >= 0.5).astype(int)  # Fixed threshold
```

---

## Recommended Next Steps

1. **Quick fix**: Run with `use_indexed_buckets=False` and see if AUC improves
2. **If AUC still ~0.58**: The problem is upstream (slicing/token signal lost)
3. **If AUC improves to 0.65+**: The indexed buckets were the main issue

### Commands to Run

```bash
# Edit config in preprocessing_full.py
"use_indexed_buckets": false

# Re-run preprocessing  
python devign_pipeline/notebooks/preprocessing_full.py

# Re-run training with focal loss
# Edit config to use focal_alpha loss type
python devign_pipeline/notebooks/02_training.py
```

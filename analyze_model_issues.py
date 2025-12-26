"""
Script to diagnose model prediction issues:
1. Analyze probability distribution (check for collapse)
2. Run TF-IDF + LogReg sanity baseline
3. Verify SEP token handling
"""

import os
import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report


DATA_DIR = "F:/Work/C Vul Devign/Dataset/devign_final"
DEBUG_DIR = os.path.join(DATA_DIR, "debug")


def load_debug_samples(split: str, max_samples: int = 5000):
    """Load tokenized samples from debug files."""
    path = os.path.join(DEBUG_DIR, f"{split}.slices_tokens.jsonl")
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def analyze_sep_tokens():
    """Check SEP token consistency."""
    print("=" * 60)
    print("1. ANALYZING SEP TOKEN CONSISTENCY")
    print("=" * 60)
    
    train_samples = load_debug_samples("train", max_samples=1000)
    
    sep_in_sliced_code = 0
    sep_in_tokens = 0
    bracket_sep_in_sliced = 0
    
    for sample in train_samples:
        sliced_code = sample.get('sliced_code', '')
        tokens = sample.get('tokens', [])
        
        if '[SEP]' in sliced_code:
            bracket_sep_in_sliced += 1
        if 'SEP' in tokens:
            sep_in_tokens += 1
        if 'SEP' in sliced_code and '[SEP]' not in sliced_code:
            sep_in_sliced_code += 1
    
    print(f"Samples analyzed: {len(train_samples)}")
    print(f"[SEP] in sliced_code: {bracket_sep_in_sliced}")
    print(f"SEP in tokens: {sep_in_tokens}")
    print(f"SEP (without brackets) in sliced_code: {sep_in_sliced_code}")
    
    if bracket_sep_in_sliced > 0 and sep_in_tokens > 0:
        print("[OK] SEP token pipeline appears CORRECT: [SEP] in code -> SEP in tokens")
    else:
        print("[WARNING] SEP token handling may be broken!")
    
    # Check a sample
    print("\nSample check (first sample with SEP):")
    for sample in train_samples[:10]:
        if 'SEP' in sample.get('tokens', []):
            print(f"  sliced_code snippet: ...{sample['sliced_code'][:200]}...")
            print(f"  tokens (first 30): {sample['tokens'][:30]}")
            break


def analyze_token_distribution():
    """Analyze token distribution and unique tokens per sample."""
    print("\n" + "=" * 60)
    print("2. ANALYZING TOKEN DISTRIBUTION")
    print("=" * 60)
    
    train_samples = load_debug_samples("train", max_samples=5000)
    
    unique_per_sample = []
    total_tokens_per_sample = []
    all_tokens = []
    
    for sample in train_samples:
        tokens = sample.get('tokens', [])
        unique_per_sample.append(len(set(tokens)))
        total_tokens_per_sample.append(len(tokens))
        all_tokens.extend(tokens)
    
    print(f"Total samples: {len(train_samples)}")
    print(f"Avg tokens per sample: {np.mean(total_tokens_per_sample):.1f}")
    print(f"Avg UNIQUE tokens per sample: {np.mean(unique_per_sample):.1f}")
    print(f"Std unique tokens: {np.std(unique_per_sample):.1f}")
    print(f"Min/Max unique: {min(unique_per_sample)} / {max(unique_per_sample)}")
    
    token_counts = Counter(all_tokens)
    print(f"\nTotal unique tokens in corpus: {len(token_counts)}")
    print("Top 20 most common tokens:")
    for token, count in token_counts.most_common(20):
        print(f"  {token}: {count}")


def analyze_class_token_differences():
    """Check if vuln and safe samples have different token patterns."""
    print("\n" + "=" * 60)
    print("3. ANALYZING CLASS-SPECIFIC TOKEN PATTERNS")
    print("=" * 60)
    
    train_samples = load_debug_samples("train", max_samples=5000)
    
    vuln_tokens = []
    safe_tokens = []
    
    for sample in train_samples:
        tokens = sample.get('tokens', [])
        label = sample.get('label', 0)
        if label == 1:
            vuln_tokens.extend(tokens)
        else:
            safe_tokens.extend(tokens)
    
    vuln_counts = Counter(vuln_tokens)
    safe_counts = Counter(safe_tokens)
    
    # Normalize counts
    vuln_total = sum(vuln_counts.values())
    safe_total = sum(safe_counts.values())
    
    # Find tokens with largest difference
    all_tokens_set = set(vuln_counts.keys()) | set(safe_counts.keys())
    
    diff_scores = []
    for token in all_tokens_set:
        vuln_freq = vuln_counts.get(token, 0) / vuln_total if vuln_total > 0 else 0
        safe_freq = safe_counts.get(token, 0) / safe_total if safe_total > 0 else 0
        diff = vuln_freq - safe_freq
        diff_scores.append((token, diff, vuln_freq, safe_freq))
    
    diff_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Vuln samples: {sum(1 for s in train_samples if s['label']==1)}")
    print(f"Safe samples: {sum(1 for s in train_samples if s['label']==0)}")
    
    print("\nTokens MORE frequent in VULNERABLE code:")
    for token, diff, vf, sf in diff_scores[:15]:
        if diff > 0:
            print(f"  {token}: +{diff*100:.3f}% (vuln: {vf*100:.2f}%, safe: {sf*100:.2f}%)")
    
    print("\nTokens MORE frequent in SAFE code:")
    for token, diff, vf, sf in diff_scores[::-1][:15]:
        if diff < 0:
            print(f"  {token}: {diff*100:.3f}% (vuln: {vf*100:.2f}%, safe: {sf*100:.2f}%)")


def run_tfidf_baseline():
    """Run TF-IDF + Logistic Regression baseline."""
    print("\n" + "=" * 60)
    print("4. TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    
    print("Loading data...")
    train_samples = load_debug_samples("train", max_samples=10000)
    val_samples = load_debug_samples("val", max_samples=3000)
    test_samples = load_debug_samples("test", max_samples=3000)
    
    # Convert to text
    def samples_to_text(samples):
        texts = [' '.join(s['tokens']) for s in samples]
        labels = [s['label'] for s in samples]
        return texts, labels
    
    train_texts, train_labels = samples_to_text(train_samples)
    val_texts, val_labels = samples_to_text(val_samples)
    test_texts, test_labels = samples_to_text(test_samples)
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    print(f"Train class dist: {sum(train_labels)} vuln / {len(train_labels)-sum(train_labels)} safe")
    
    # TF-IDF
    print("\nFitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"TF-IDF features: {X_train.shape[1]}")
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=1.0,
        solver='lbfgs'
    )
    clf.fit(X_train, train_labels)
    
    # Evaluate
    for split_name, X, y in [('Val', X_val, val_labels), ('Test', X_test, test_labels)]:
        probs = clf.predict_proba(X)[:, 1]
        preds = clf.predict(X)
        
        auc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        
        print(f"\n{split_name} Results:")
        print(f"  AUC:       {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
    
    # Feature importance
    print("\nTop positive features (indicate VULNERABLE):")
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-15:][::-1]
    for idx in top_pos_idx:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
    
    print("\nTop negative features (indicate SAFE):")
    top_neg_idx = np.argsort(coefs)[:15]
    for idx in top_neg_idx:
        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
    
    return auc


def check_vocab_coverage():
    """Check how many tokens are in vocab vs UNK."""
    print("\n" + "=" * 60)
    print("5. VOCAB COVERAGE CHECK")
    print("=" * 60)
    
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"Vocab size: {len(vocab)}")
    
    train_samples = load_debug_samples("train", max_samples=2000)
    
    in_vocab = 0
    out_vocab = 0
    out_vocab_tokens = Counter()
    
    for sample in train_samples:
        for token in sample['tokens']:
            if token in vocab:
                in_vocab += 1
            else:
                out_vocab += 1
                out_vocab_tokens[token] += 1
    
    total = in_vocab + out_vocab
    print(f"Tokens in vocab: {in_vocab} ({in_vocab/total*100:.2f}%)")
    print(f"Tokens NOT in vocab (UNK): {out_vocab} ({out_vocab/total*100:.2f}%)")
    
    if out_vocab > 0:
        print(f"\nTop OOV tokens:")
        for token, count in out_vocab_tokens.most_common(20):
            print(f"  {token}: {count}")


if __name__ == "__main__":
    analyze_sep_tokens()
    analyze_token_distribution()
    analyze_class_token_differences()
    check_vocab_coverage()
    
    print("\n" + "=" * 60)
    print("RUNNING SANITY BASELINE...")
    print("=" * 60)
    baseline_auc = run_tfidf_baseline()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TF-IDF + LogReg Baseline AUC: {baseline_auc:.4f}")
    print(f"BiGRU Model AUC:              ~0.58")
    print("\nIf baseline AUC ≈ BiGRU AUC:")
    print("  → Problem is in DATA/REPRESENTATION (tokens lack discriminative power)")
    print("\nIf baseline AUC > BiGRU AUC:")
    print("  → Problem is in MODEL/TRAINING (architecture or training issues)")
    print("\nIf baseline AUC < 0.55:")
    print("  → Problem is FUNDAMENTAL: labels may be noisy or task too hard")

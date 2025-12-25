# Devign CodeBERT Training - HierarchicalBiGRU with CodeBERT Embeddings
# Target: F1 > 0.80, AUC > 0.90
import os, sys, json, random, numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ===== INSTALL TRANSFORMERS IF NEEDED =====
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    os.system('pip install transformers -q')
    from transformers import AutoTokenizer, AutoModel

# ===== CONFIG PATHS =====
if os.path.exists('/kaggle'):
    DATA_DIR = '/kaggle/input/devign'  # Parquet files
    OUTPUT_DIR = '/kaggle/working/output'
    CACHE_DIR = '/kaggle/working/cache'
else:
    DATA_DIR = '/media/tuananh/새 볼륨/DACNANM/Devign/C-Vul-Devign/Dataset/devign slice'
    OUTPUT_DIR = './output_codebert'
    CACHE_DIR = './cache'

MODEL_DIR = f'{OUTPUT_DIR}/models'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# ===== CONFIGURATION =====
CODEBERT_MODEL = "microsoft/codebert-base"  # or "microsoft/graphcodebert-base"
MAX_LENGTH = 512  # CodeBERT max tokens
BATCH_SIZE = 8    # Smaller due to CodeBERT memory
NUM_EPOCHS = 10   # Fewer epochs needed with pretrained
LEARNING_RATE = 2e-5  # Lower LR for fine-tuning
FREEZE_CODEBERT_EPOCHS = 2  # Freeze CodeBERT for first N epochs
NUM_SEEDS = 1     # Set to 3 for ensemble

# ===== LOAD RAW DATA =====
print("Loading raw data...")

def load_data(data_dir):
    """Load raw Devign data from various formats."""
    data_dir = Path(data_dir)
    
    # Try HuggingFace sharded format (train-00000-of-00001-*.parquet)
    train_shards = sorted(data_dir.glob('train-*.parquet'))
    if train_shards:
        print("Loading HuggingFace sharded parquet format...")
        train_df = pd.concat([pd.read_parquet(f) for f in train_shards], ignore_index=True)
        val_shards = sorted(data_dir.glob('validation-*.parquet'))
        val_df = pd.concat([pd.read_parquet(f) for f in val_shards], ignore_index=True) if val_shards else None
        test_shards = sorted(data_dir.glob('test-*.parquet'))
        test_df = pd.concat([pd.read_parquet(f) for f in test_shards], ignore_index=True) if test_shards else None
        return train_df, val_df, test_df
    
    # Try JSONL format
    if (data_dir / 'train.jsonl').exists():
        print("Loading JSONL format...")
        train_df = pd.read_json(data_dir / 'train.jsonl', lines=True)
        val_df = pd.read_json(data_dir / 'valid.jsonl', lines=True) if (data_dir / 'valid.jsonl').exists() else pd.read_json(data_dir / 'val.jsonl', lines=True)
        test_df = pd.read_json(data_dir / 'test.jsonl', lines=True)
        return train_df, val_df, test_df
    
    # Try JSON format
    if (data_dir / 'train.json').exists():
        print("Loading JSON format...")
        train_df = pd.read_json(data_dir / 'train.json')
        val_df = pd.read_json(data_dir / 'valid.json') if (data_dir / 'valid.json').exists() else pd.read_json(data_dir / 'val.json')
        test_df = pd.read_json(data_dir / 'test.json')
        return train_df, val_df, test_df
    
    # Try Parquet format
    if (data_dir / 'train.parquet').exists():
        print("Loading Parquet format...")
        train_df = pd.read_parquet(data_dir / 'train.parquet')
        val_df = pd.read_parquet(data_dir / 'val.parquet')
        test_df = pd.read_parquet(data_dir / 'test.parquet')
        return train_df, val_df, test_df
    
    # List available files for debugging
    files = list(data_dir.glob('*'))
    print(f"Available files in {data_dir}:")
    for f in files[:20]:
        print(f"  {f.name}")
    
    return None, None, None

train_df, val_df, test_df = load_data(DATA_DIR)

if train_df is None:
    print(f"ERROR: No valid data files found in {DATA_DIR}")
    print("Expected: train.jsonl/json/parquet, val.jsonl/json/parquet, test.jsonl/json/parquet")
    sys.exit(1)

print(f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# ===== CODEBERT TOKENIZER =====
print(f"Loading CodeBERT tokenizer: {CODEBERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL, cache_dir=CACHE_DIR)

# ===== DATASET =====
class CodeBERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract codes and labels from DataFrame
        if 'func' in df.columns:
            self.codes = df['func'].tolist()
        elif 'code' in df.columns:
            self.codes = df['code'].tolist()
        else:
            self.codes = df['function'].tolist()
        
        self.labels = df['target'].values.astype(np.int64)
        print(f"  Dataset: {len(self.codes)} samples, pos={np.sum(self.labels)}, neg={np.sum(1-self.labels)}")
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        
        # Tokenize with CodeBERT
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("Creating datasets...")
train_ds = CodeBERTDataset(train_df, tokenizer, MAX_LENGTH)
val_ds = CodeBERTDataset(val_df, tokenizer, MAX_LENGTH)
test_ds = CodeBERTDataset(test_df, tokenizer, MAX_LENGTH)

n_neg, n_pos = np.sum(train_ds.labels==0), np.sum(train_ds.labels==1)
print(f"Classes: neg={n_neg}, pos={n_pos}, ratio={n_neg/n_pos:.2f}")

# ===== MODEL: CodeBERT + BiGRU =====
class CodeBERTBiGRU(nn.Module):
    def __init__(self, model_name=CODEBERT_MODEL, hidden_dim=256, num_layers=1, dropout=0.3):
        super().__init__()
        
        # Load CodeBERT
        self.codebert = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.bert_dim = self.codebert.config.hidden_size  # 768
        
        # BiGRU on top of CodeBERT
        self.bigru = nn.GRU(
            self.bert_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        self._frozen = False
    
    def freeze_codebert(self):
        """Freeze CodeBERT parameters."""
        for param in self.codebert.parameters():
            param.requires_grad = False
        self._frozen = True
        print("CodeBERT frozen")
    
    def unfreeze_codebert(self):
        """Unfreeze CodeBERT parameters."""
        for param in self.codebert.parameters():
            param.requires_grad = True
        self._frozen = False
        print("CodeBERT unfrozen")
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # CodeBERT encoding
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, 768]
        
        # BiGRU
        gru_output, _ = self.bigru(sequence_output)  # [B, L, hidden*2]
        
        # Attention pooling
        attn_scores = self.attention(gru_output)  # [B, L, 1]
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (gru_output * attn_weights).sum(dim=1)  # [B, hidden*2]
        
        # Classification
        logits = self.classifier(context)
        return logits

# Alternative: Simple CodeBERT + MLP (lighter)
class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name=CODEBERT_MODEL, dropout=0.3):
        super().__init__()
        
        self.codebert = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.bert_dim = self.codebert.config.hidden_size  # 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.bert_dim),
            nn.Dropout(dropout),
            nn.Linear(self.bert_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self._frozen = False
    
    def freeze_codebert(self):
        for param in self.codebert.parameters():
            param.requires_grad = False
        self._frozen = True
        print("CodeBERT frozen")
    
    def unfreeze_codebert(self):
        for param in self.codebert.parameters():
            param.requires_grad = True
        self._frozen = False
        print("CodeBERT unfrozen")
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token or mean pooling
        # Option 1: CLS token
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # Option 2: Mean pooling (often better)
        # mask = attention_mask.unsqueeze(-1).float()
        # cls_output = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        
        logits = self.classifier(cls_output)
        return logits

# ===== TRAINING FUNCTIONS =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, scaler, epoch, freeze_epochs):
    model.train()
    
    # Freeze/unfreeze CodeBERT based on epoch
    if epoch <= freeze_epochs and not model._frozen:
        model.freeze_codebert()
    elif epoch > freeze_epochs and model._frozen:
        model.unfreeze_codebert()
    
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    
    for batch in tqdm(loader, desc=f"Train Epoch {epoch}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            logits = model(**batch)
            loss = criterion(logits, batch['labels'])
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        probs = F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []
    
    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        with autocast(device_type='cuda'):
            logits = model(**batch)
            loss = criterion(logits, batch['labels'])
        
        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    
    # Find optimal threshold
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.25, 0.75, 0.01):
        f1 = f1_score(all_labels, (all_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    opt_preds = (all_probs >= best_t).astype(int)
    
    return {
        'loss': total_loss / len(loader),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'opt_f1': best_f1,
        'opt_prec': precision_score(all_labels, opt_preds),
        'opt_rec': recall_score(all_labels, opt_preds),
        'opt_t': best_t,
        'labels': all_labels,
        'probs': all_probs
    }

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    
    for batch in tqdm(loader, desc="Predict"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with autocast(device_type='cuda'):
            logits = model(**batch)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch['labels'].cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)

def train_model(seed, train_ds, val_ds, n_neg, n_pos):
    print(f"\n{'#'*60}\nTraining with seed {seed}\n{'#'*60}")
    set_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    # Choose model architecture
    # model = CodeBERTClassifier(CODEBERT_MODEL, dropout=0.3).to(DEVICE)  # Simpler
    model = CodeBERTBiGRU(CODEBERT_MODEL, hidden_dim=256, dropout=0.3).to(DEVICE)  # Hybrid
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss with class weights
    pos_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    
    # Optimizer with different LR for CodeBERT vs head
    codebert_params = list(model.codebert.parameters())
    head_params = [p for n, p in model.named_parameters() if 'codebert' not in n]
    
    optimizer = optim.AdamW([
        {'params': codebert_params, 'lr': LEARNING_RATE},
        {'params': head_params, 'lr': LEARNING_RATE * 10}  # Higher LR for head
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()
    
    best_f1, patience = 0, 0
    PATIENCE = 3
    model_path = f'{MODEL_DIR}/codebert_best_seed{seed}.pt'
    
    history = {
        'train_loss': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_f1': [], 'val_auc': [],
        'val_opt_f1': [], 'val_prec': [], 'val_rec': []
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{NUM_EPOCHS}")
        
        train_m = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, FREEZE_CODEBERT_EPOCHS)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Train: loss={train_m['loss']:.4f}, F1={train_m['f1']:.4f}, AUC={train_m['auc']:.4f}")
        print(f"Val: loss={val_m['loss']:.4f}, F1={val_m['f1']:.4f}, AUC={val_m['auc']:.4f}")
        print(f"Val Opt: F1={val_m['opt_f1']:.4f}, Prec={val_m['opt_prec']:.4f}, Rec={val_m['opt_rec']:.4f}, t={val_m['opt_t']:.2f}")
        
        # Save history
        history['train_loss'].append(train_m['loss'])
        history['train_f1'].append(train_m['f1'])
        history['train_auc'].append(train_m['auc'])
        history['val_loss'].append(val_m['loss'])
        history['val_f1'].append(val_m['f1'])
        history['val_auc'].append(val_m['auc'])
        history['val_opt_f1'].append(val_m['opt_f1'])
        history['val_prec'].append(val_m['opt_prec'])
        history['val_rec'].append(val_m['opt_rec'])
        
        if val_m['opt_f1'] > best_f1:
            best_f1 = val_m['opt_f1']
            patience = 0
            torch.save(model.state_dict(), model_path)
            print(f"★ Best F1: {best_f1:.4f}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(model_path, weights_only=False))
    return model, best_f1, history

# ===== MAIN TRAINING =====
print(f"\n{'='*60}")
print(f"Training CodeBERT + BiGRU Model")
print(f"{'='*60}")

models = []
val_probs_list = []
test_probs_list = []
all_histories = []

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

for seed_idx in range(NUM_SEEDS):
    seed = 42 + seed_idx * 1000
    model, best_f1, history = train_model(seed, train_ds, val_ds, n_neg, n_pos)
    models.append(model)
    all_histories.append(history)
    
    _, val_probs = get_predictions(model, val_loader)
    _, test_probs = get_predictions(model, test_loader)
    val_probs_list.append(val_probs)
    test_probs_list.append(test_probs)

# ===== ENSEMBLE & EVALUATION =====
print(f"\n{'='*60}")
print(f"Ensemble Evaluation ({NUM_SEEDS} models)")
print(f"{'='*60}")

val_labels = val_ds.labels
test_labels = test_ds.labels

val_probs_ens = np.mean(val_probs_list, axis=0)
test_probs_ens = np.mean(test_probs_list, axis=0)

# Calibration
print("Applying isotonic calibration...")
calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrator.fit(val_probs_ens, val_labels)
val_probs_cal = calibrator.predict(val_probs_ens)
test_probs_cal = calibrator.predict(test_probs_ens)

# Find optimal threshold
best_f1, best_t = 0, 0.5
for t in np.arange(0.25, 0.75, 0.01):
    f1 = f1_score(val_labels, (val_probs_cal >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Validation: OptF1={best_f1:.4f} at t={best_t:.2f}")

# Test evaluation
test_preds_cal = (test_probs_cal >= best_t).astype(int)
test_preds_05 = (test_probs_cal >= 0.5).astype(int)

test_f1_05 = f1_score(test_labels, test_preds_05)
test_f1_opt = f1_score(test_labels, test_preds_cal)
test_prec = precision_score(test_labels, test_preds_cal)
test_rec = recall_score(test_labels, test_preds_cal)
test_auc = roc_auc_score(test_labels, test_probs_cal)

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS (CodeBERT + BiGRU)")
print(f"{'='*60}")
print(f"Test F1 (t=0.5): {test_f1_05:.4f}")
print(f"Test OptF1 (t={best_t:.2f}): {test_f1_opt:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# ===== SAVE RESULTS =====
results = {
    'model': 'CodeBERT + BiGRU',
    'codebert_model': CODEBERT_MODEL,
    'num_seeds': NUM_SEEDS,
    'optimal_threshold': float(best_t),
    'test_f1_05': float(test_f1_05),
    'test_opt_f1': float(test_f1_opt),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'test_auc': float(test_auc)
}

with open(f'{OUTPUT_DIR}/codebert_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save config.json
config_data = {
    'model_type': 'CodeBERT + BiGRU',
    'codebert_model': CODEBERT_MODEL,
    'max_length': MAX_LENGTH,
    'hidden_dim': HIDDEN_DIM,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'num_epochs': NUM_EPOCHS,
    'freeze_codebert_epochs': FREEZE_CODEBERT_EPOCHS,
    'num_seeds': NUM_SEEDS,
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df)
}
with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(config_data, f, indent=2)

# Save training history
history_data = {
    'all_histories': all_histories,
    'best_f1_scores': [h['best_val_f1'] for h in all_histories] if all_histories else []
}
with open(f'{OUTPUT_DIR}/training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)

# Save predictions
predictions_data = {
    'test_labels': test_labels.tolist(),
    'test_probs': test_probs_cal.tolist(),
    'test_preds_05': test_preds_05.tolist(),
    'test_preds_opt': test_preds_cal.tolist(),
    'optimal_threshold': float(best_t)
}
with open(f'{OUTPUT_DIR}/predictions.json', 'w') as f:
    json.dump(predictions_data, f, indent=2)

# Save classification report
from sklearn.metrics import classification_report
report = classification_report(test_labels, test_preds_cal, target_names=['Non-Vulnerable', 'Vulnerable'], output_dict=True)
with open(f'{OUTPUT_DIR}/classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Save vocab.json (CodeBERT tokenizer info)
vocab_data = {
    'tokenizer': CODEBERT_MODEL,
    'vocab_size': tokenizer.vocab_size,
    'max_length': MAX_LENGTH,
    'special_tokens': {
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'cls_token': tokenizer.cls_token,
        'sep_token': tokenizer.sep_token,
        'mask_token': tokenizer.mask_token,
        'pad_token_id': tokenizer.pad_token_id,
        'unk_token_id': tokenizer.unk_token_id,
        'cls_token_id': tokenizer.cls_token_id,
        'sep_token_id': tokenizer.sep_token_id,
        'mask_token_id': tokenizer.mask_token_id
    }
}
with open(f'{OUTPUT_DIR}/vocab.json', 'w') as f:
    json.dump(vocab_data, f, indent=2)

print(f"\nSaved JSON files to {OUTPUT_DIR}:")
print(f"  - codebert_results.json")
print(f"  - config.json")
print(f"  - training_history.json")
print(f"  - predictions.json")
print(f"  - classification_report.json")
print(f"  - vocab.json")

# ===== PLOTS =====
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm1 = confusion_matrix(test_labels, test_preds_05)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Vuln', 'Vuln'], yticklabels=['Non-Vuln', 'Vuln'])
axes[0].set_title(f'Confusion Matrix (t=0.5)\nF1={test_f1_05:.4f}')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

cm2 = confusion_matrix(test_labels, test_preds_cal)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Non-Vuln', 'Vuln'], yticklabels=['Non-Vuln', 'Vuln'])
axes[1].set_title(f'Confusion Matrix (t={best_t:.2f})\nOptF1={test_f1_opt:.4f}')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/confusion_matrix_codebert.png', dpi=150)
plt.show()

# ROC & PR Curves
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fpr, tpr, _ = roc_curve(test_labels, test_probs_cal)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - CodeBERT + BiGRU', fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

precision, recall, _ = precision_recall_curve(test_labels, test_probs_cal)
pr_auc = auc(recall, precision)
axes[1].plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve - CodeBERT + BiGRU', fontweight='bold')
axes[1].legend(loc='lower left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/roc_pr_codebert.png', dpi=150)
plt.show()

# Metrics Summary
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['F1 (t=0.5)', 'F1 (optimal)', 'Precision', 'Recall', 'AUC']
metrics_values = [test_f1_05, test_f1_opt, test_prec, test_rec, test_auc]
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']

bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black')
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title(f'CodeBERT + BiGRU Test Metrics (t={best_t:.2f})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/metrics_codebert.png', dpi=150)
plt.show()

print(f"\n{'='*60}")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"{'='*60}")

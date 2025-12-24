# %% [markdown]
# # CodeBERT Vulnerability Detection Training on Kaggle (Dual T4 GPUs)
# 
# This notebook trains CodeBERT for C/C++ vulnerability detection using Kaggle's dual T4 GPU setup.
# 
# **Hardware:** 2x NVIDIA Tesla T4 (16GB VRAM each)  
# **Dataset:** devign_final (from hdiii/create-devign-dataset)
# **Expected Training Time:** ~2-3 hours for full dataset

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Install required packages
!pip install -q transformers==4.36.0 accelerate scikit-learn tqdm

# %% [markdown]
# ## 2. Imports and Setup

# %%
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# %% [markdown]
# ## 3. Check GPU Availability

# %%
def check_gpu_setup():
    """Check and display GPU configuration"""
    print("=" * 60)
    print("GPU CONFIGURATION CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! Please enable GPU in Kaggle settings.")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"\n📊 GPU {i}: {props.name}")
        print(f"   - Total Memory: {memory_gb:.1f} GB")
        print(f"   - Compute Capability: {props.major}.{props.minor}")
    
    print("=" * 60)
    return True

gpu_available = check_gpu_setup()

# %% [markdown]
# ## 4. Dataset Paths - devign_final

# %%
# Dataset: devign_final from hdiii/create-devign-dataset
# Structure:
#   processed/
#     ├── train.npz, val.npz, test.npz (tokenized for BiGRU - NOT used)
#     ├── train_tokens.jsonl, val_tokens.jsonl, test_tokens.jsonl (raw tokens)
#     ├── train_token_with_id.jsonl, val_token_with_id.jsonl, test_token_with_id.jsonl
#     └── config.json

KAGGLE_INPUT_PATH = "/kaggle/input"
KAGGLE_OUTPUT_PATH = "/kaggle/working"

# Dataset name on Kaggle
DATASET_NAME = "devign-final"
DATA_PATH = os.path.join(KAGGLE_INPUT_PATH, DATASET_NAME, "processed")

# Fallback to local for testing
if not os.path.exists(DATA_PATH):
    # Try alternative paths
    alt_paths = [
        "./Dataset/processed",
        "../Dataset/processed",
        "/kaggle/input/devign-final/processed"
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            DATA_PATH = alt
            break
    
    KAGGLE_OUTPUT_PATH = "./output"
    os.makedirs(KAGGLE_OUTPUT_PATH, exist_ok=True)

print(f"📂 Data path: {DATA_PATH}")
print(f"📂 Output path: {KAGGLE_OUTPUT_PATH}")

# List available files
if os.path.exists(DATA_PATH):
    print(f"\n📄 Available files:")
    for f in sorted(os.listdir(DATA_PATH)):
        size_kb = os.path.getsize(os.path.join(DATA_PATH, f)) / 1024
        print(f"   - {f} ({size_kb:.1f} KB)")

# %% [markdown]
# ## 5. Configuration

# %%
class Config:
    """Training configuration for CodeBERT on dual T4 GPUs"""
    
    # Model
    model_name = "microsoft/codebert-base"
    max_length = 512
    num_labels = 2
    
    # Training - Optimized for 2x T4
    batch_size_per_gpu = 16
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = batch_size_per_gpu * max(num_gpus, 1)
    
    # Hyperparameters
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 10
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    
    # Mixed precision
    use_fp16 = True
    
    # Logging
    log_interval = 100
    
    # Early stopping
    patience = 3
    
    # Paths
    output_dir = KAGGLE_OUTPUT_PATH
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    @classmethod
    def display(cls):
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Model: {cls.model_name}")
        print(f"Max sequence length: {cls.max_length}")
        print(f"Batch size per GPU: {cls.batch_size_per_gpu}")
        print(f"Number of GPUs: {cls.num_gpus}")
        print(f"Effective batch size: {cls.effective_batch_size}")
        print(f"Learning rate: {cls.learning_rate}")
        print(f"Epochs: {cls.num_epochs}")
        print(f"FP16 training: {cls.use_fp16}")
        print("=" * 60)

Config.display()
os.makedirs(Config.checkpoint_dir, exist_ok=True)

# %% [markdown]
# ## 6. Data Loading from devign_final

# %%
class CodeBERTDataset(Dataset):
    """Dataset for CodeBERT training from devign_final JSONL files
    
    Expected format (from devign_final):
    {
        "sample_id": 0,
        "label": 0,
        "original_code": "int func(...) {...}",
        "sliced_code": "int func(...) ...",
        "tokens": ["int", "FUNC", ...]
    }
    """
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, 
                 use_sliced: bool = False):
        """
        Args:
            jsonl_path: Path to JSONL file
            tokenizer: CodeBERT tokenizer
            max_length: Max sequence length (512)
            use_sliced: If True, use sliced_code; else use original_code
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_sliced = use_sliced
        self.data = []
        
        print(f"Loading {jsonl_path}...")
        print(f"Using {'sliced_code' if use_sliced else 'original_code'}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
        
        print(f"Loaded {len(self.data):,} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get code - prioritize based on use_sliced flag
        if self.use_sliced and 'sliced_code' in item:
            code = item['sliced_code']
        elif 'original_code' in item:
            code = item['original_code']
        elif 'sliced_code' in item:
            code = item['sliced_code']
        elif 'func' in item:
            code = item['func']
        else:
            # Fallback: join tokens
            code = ' '.join(item.get('tokens', []))
        
        # Get label
        label = item.get('label', item.get('target', 0))
        
        # Clean code
        code = self._clean_code(code)
        
        # Tokenize with CodeBERT tokenizer
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
    
    @staticmethod
    def _clean_code(code: str) -> str:
        """Remove comments and normalize whitespace"""
        import re
        # Remove C-style comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove C++ style comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        # Normalize whitespace
        code = ' '.join(code.split())
        return code


class RawCodeDataset(Dataset):
    """Dataset that loads raw code from NPZ + separate JSONL for raw text"""
    
    def __init__(self, npz_path: str, jsonl_path: str, tokenizer, max_length: int = 512):
        """
        Load labels from NPZ, raw code from JSONL
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load labels from NPZ
        print(f"Loading labels from {npz_path}...")
        npz_data = np.load(npz_path)
        self.labels = npz_data['labels']
        print(f"Loaded {len(self.labels):,} labels")
        
        # Load raw code from JSONL
        print(f"Loading code from {jsonl_path}...")
        self.codes = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading code"):
                if line.strip():
                    item = json.loads(line)
                    if 'tokens' in item:
                        code = ' '.join(item['tokens'])
                    else:
                        code = item.get('func', item.get('code', ''))
                    self.codes.append(code)
        
        # Verify matching lengths
        if len(self.codes) != len(self.labels):
            print(f"⚠️ Mismatch: {len(self.codes)} codes vs {len(self.labels)} labels")
            min_len = min(len(self.codes), len(self.labels))
            self.codes = self.codes[:min_len]
            self.labels = self.labels[:min_len]
        
        print(f"Dataset ready: {len(self.codes):,} samples")
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = self.codes[idx]
        label = int(self.labels[idx])
        
        # Clean and tokenize
        code = self._clean_code(code)
        
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
    
    @staticmethod
    def _clean_code(code: str) -> str:
        import re
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = ' '.join(code.split())
        return code


def prepare_dataloaders(tokenizer, config, data_path, use_sliced=False):
    """Prepare dataloaders from devign_final dataset
    
    Args:
        use_sliced: If True, use sliced_code; else use original_code
                    Set to False for fair evaluation (no label leakage)
    """
    
    # Find files - prioritize *_tokens.jsonl (has original_code + sliced_code)
    train_jsonl = os.path.join(data_path, "train_tokens.jsonl")
    val_jsonl = os.path.join(data_path, "val_tokens.jsonl")
    test_jsonl = os.path.join(data_path, "test_tokens.jsonl")
    
    print(f"📄 Train: {train_jsonl}")
    print(f"📄 Val: {val_jsonl}")
    print(f"📄 Test: {test_jsonl}")
    print(f"📄 Using: {'sliced_code' if use_sliced else 'original_code'}")
    
    # Create datasets
    train_dataset = CodeBERTDataset(train_jsonl, tokenizer, config.max_length, use_sliced)
    val_dataset = CodeBERTDataset(val_jsonl, tokenizer, config.max_length, use_sliced)
    test_dataset = CodeBERTDataset(test_jsonl, tokenizer, config.max_length, use_sliced)
    
    # Show label distribution
    train_labels = [d.get('label', d.get('target', 0)) for d in train_dataset.data]
    print(f"\n📊 Train label distribution:")
    print(f"   Non-vulnerable (0): {train_labels.count(0):,}")
    print(f"   Vulnerable (1): {train_labels.count(1):,}")
    
    # Calculate class weights for imbalanced data
    n_neg = train_labels.count(0)
    n_pos = train_labels.count(1)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"   Pos weight: {pos_weight:.2f}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.effective_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.effective_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.effective_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, pos_weight

# %% [markdown]
# ## 7. Model Definition

# %%
class CodeBERTClassifier(nn.Module):
    """CodeBERT-based classifier for vulnerability detection"""
    
    def __init__(self, model_name, num_labels=2, dropout=0.1, pos_weight=None):
        super().__init__()
        
        self.codebert = RobertaModel.from_pretrained(model_name)
        hidden_size = self.codebert.config.hidden_size
        
        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Class weight for imbalanced data
        self.pos_weight = pos_weight
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            if self.pos_weight is not None:
                weight = torch.tensor([1.0, self.pos_weight], device=logits.device)
                loss_fn = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def initialize_model(config, pos_weight=None):
    """Initialize model with DataParallel for multi-GPU"""
    
    print("🔄 Loading CodeBERT model...")
    model = CodeBERTClassifier(
        model_name=config.model_name,
        num_labels=config.num_labels,
        pos_weight=pos_weight
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"✅ Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    return model, device

# Load tokenizer
print("🔄 Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(Config.model_name)
print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")

# %% [markdown]
# ## 8. Training Loop

# %%
def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]",
        leave=True
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if config.use_fp16:
            with autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                if isinstance(model, nn.DataParallel):
                    loss = loss.mean()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate(model, eval_loader, device, config, desc="Eval"):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    progress_bar = tqdm(eval_loader, desc=desc, leave=True)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if config.use_fp16:
                with autocast():
                    outputs = model(input_ids, attention_mask, labels)
            else:
                outputs = model(input_ids, attention_mask, labels)
            
            loss = outputs['loss']
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs['logits'], dim=-1)
            preds = outputs['logits'].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    return {
        'loss': total_loss / len(eval_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train(model, train_loader, val_loader, device, config):
    """Full training loop"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler() if config.use_fp16 else None
    
    best_f1 = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print(f"Total steps: {total_steps:,}, Warmup: {warmup_steps:,}")
    print("=" * 60 + "\n")
    
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config, epoch
        )
        history['train'].append(train_metrics)
        
        val_metrics = evaluate(
            model, val_loader, device, config, 
            desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"
        )
        history['val'].append(val_metrics)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            save_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'best_f1': best_f1,
            }, save_path)
            print(f"   ✅ New best model! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{config.patience}")
        
        if patience_counter >= config.patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
        
        print()
    
    return history, best_f1

# %% [markdown]
# ## 9. Run Training

# %%
# === CONFIGURATION ===
# Set USE_SLICED = True to use sliced_code (higher signal but potential label leakage)
# Set USE_SLICED = False to use original_code (fair evaluation, no leakage)
USE_SLICED = False  # Recommended: False for fair comparison

# Prepare data
train_loader, val_loader, test_loader, pos_weight = prepare_dataloaders(
    tokenizer, Config, DATA_PATH, use_sliced=USE_SLICED
)

# Initialize model with class weights
model, device = initialize_model(Config, pos_weight=pos_weight)

# Train
history, best_f1 = train(model, train_loader, val_loader, device, Config)

# %% [markdown]
# ## 10. Final Evaluation

# %%
def display_final_results(model, test_loader, device, config):
    """Load best model and evaluate on test set"""
    
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        print("🔄 Loading best model...")
        checkpoint = torch.load(checkpoint_path)
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model with F1: {checkpoint['best_f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_metrics = evaluate(model, test_loader, device, config, desc="Testing")
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print(f"\n📊 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Non-Vuln  Vuln")
    print(f"   Actual Non-Vuln   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"   Actual Vuln       {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    print("=" * 60)
    return test_metrics

test_metrics = display_final_results(model, test_loader, device, Config)

# %% [markdown]
# ## 11. Save Model

# %%
def save_final_model(model, tokenizer, config, test_metrics):
    """Save model for deployment"""
    
    output_dir = os.path.join(config.output_dir, 'codebert_vuln_detector')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Saving to {output_dir}")
    
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    
    # Save model
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    model_to_save.codebert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save classifier
    torch.save(
        model_to_save.classifier.state_dict(),
        os.path.join(output_dir, 'classifier.bin')
    )
    
    # Save results
    results = {
        'model_name': config.model_name,
        'max_length': config.max_length,
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'roc_auc': float(test_metrics['roc_auc'])
        }
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Model saved!")
    return output_dir

output_path = save_final_model(model, tokenizer, Config, test_metrics)

# %% [markdown]
# ## 12. Inference Example

# %%
def predict_vulnerability(code, model, tokenizer, device, max_length=512):
    """Predict if code is vulnerable"""
    
    model.eval()
    
    encoding = tokenizer(
        code,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs['logits'], dim=-1)
        pred = outputs['logits'].argmax(dim=-1).item()
    
    return {
        'prediction': 'VULNERABLE' if pred == 1 else 'SAFE',
        'confidence': probs[0][pred].item(),
        'vulnerable_prob': probs[0][1].item()
    }

# Test examples
vulnerable_code = '''
void copy_input(char *user_input) {
    char buffer[64];
    strcpy(buffer, user_input);
    printf("Input: %s\\n", buffer);
}
'''

safe_code = '''
void copy_input(char *user_input) {
    char buffer[64];
    strncpy(buffer, user_input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\\0';
    printf("Input: %s\\n", buffer);
}
'''

print("=" * 60)
print("INFERENCE EXAMPLES")
print("=" * 60)

print("\n📝 Vulnerable Code:")
result1 = predict_vulnerability(vulnerable_code, model, tokenizer, device)
print(f"   {result1['prediction']} (conf: {result1['confidence']:.3f})")

print("\n📝 Safe Code:")
result2 = predict_vulnerability(safe_code, model, tokenizer, device)
print(f"   {result2['prediction']} (conf: {result2['confidence']:.3f})")

print("\n" + "=" * 60)
print("TRAINING COMPLETE! 🎉")
print("=" * 60)

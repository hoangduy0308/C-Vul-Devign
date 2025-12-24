---
name: c-vuln-devign
description: Xử lý dataset C vulnerability detection với Devign. Hỗ trợ preprocessing (AST, slicing, tokenization, vocab), training BiGRU, và inference. Dùng khi debug lỗi pipeline, tối ưu hyperparameters, hoặc làm việc với code C vulnerability.
---

# C Vulnerability Detection - Devign Pipeline

Pipeline phát hiện lỗ hổng bảo mật trong code C/C++ sử dụng BiGRU model.

## Cấu trúc dự án

```
devign_pipeline/
├── notebooks/
│   ├── prepare_data.py         # Full preprocessing pipeline (PreserveIdentifierTokenizer)
│   ├── 02_training.py          # BiGRU training với AMP, multi-GPU
│   └── 03_preprocessing_v2.py  # Preprocessing v2 (CanonicalTokenizer)
├── src/
│   ├── ast/parser.py           # Tree-sitter C/C++ parser
│   ├── slicing/
│   │   ├── slicer.py           # Backward/forward slicing với CFG/DFG
│   │   └── multi_slicer.py     # Combined multi-slice
│   ├── tokenization/
│   │   ├── tokenizer.py        # CTokenizer (tree-sitter based)
│   │   ├── hybrid_tokenizer.py # CanonicalTokenizer + HybridTokenizer
│   │   ├── preserve_tokenizer.py
│   │   └── vocab.py
│   ├── graphs/
│   │   ├── cfg.py              # Control Flow Graph builder
│   │   └── dfg.py              # Data Flow Graph builder
│   └── vuln/                   # Vulnerability features extraction
└── config/
```

## Environment

**Local:** `f:/Work/C Vul Devign/`
**Kaggle:**
```python
DATA_DIR = '/kaggle/input/devign'
WORKING_DIR = '/kaggle/working'
sys.path.insert(0, '/kaggle/input/devign-pipeline/devign_pipeline')
```

---

# PREPROCESSING WORKFLOW

## Pipeline Flow
```
Raw C code → Parse AST (tree-sitter) → Build CFG/DFG → 
Find criterion lines (dangerous APIs) → Slicing (backward/forward) → 
Tokenize → Build vocab → Vectorize → .npz files
```

## 1. AST Parsing (tree-sitter)

### Luồng chính
```python
from src.ast.parser import CFamilyParser, Language

parser = CFamilyParser(Language.C)
result = parser.parse_with_fallback(code)  # Thử C++ trước, fallback C
```

### Cấu trúc output
- `ParseResult.nodes`: List[ASTNode] - danh sách node phẳng
- `ParseResult.has_errors`: bool - có lỗi parse không
- `ParseResult.error_count`: int - số node ERROR

### Detect ngôn ngữ tự động
```python
lang = CFamilyParser.detect_language(code, project='qemu')  # Dựa vào patterns
```

**Patterns C++ được detect:** class, template, namespace, std::, nullptr, override...

### ⚠️ Lưu ý quan trọng
- `parse()` có thể return `None` khi exception → downstream phải check
- Node `text` decode UTF-8 có try/except, có thể mất text với ký tự lỗi

---

## 2. CFG Building (Control Flow Graph)

### Mục tiêu
Tạo graph block-level để truy control dependencies.

### Cách hoạt động
- Tìm control structures: `if_statement`, `while_statement`, `for_statement`, `switch_statement`...
- Cắt basic blocks theo start/end line của control nodes
- Nối edges theo if/loop/switch/goto/return/break/continue

### API
```python
from src.graphs.cfg import CFGBuilder

cfg_builder = CFGBuilder()
cfg = cfg_builder.build(parse_result)

# Check empty
if cfg is None or cfg.is_empty():
    # Fallback to window slice
```

### ⚠️ Hạn chế
- Xử lý `goto` có thể fail với label phức tạp
- Không phải full compiler IR → miss một số control flow phức tạp

---

## 3. DFG Building (Data Flow Graph)

### Mục tiêu
Tạo danh sách variable accesses + edges def-use để backward/forward slice theo data deps.

### Cách hoạt động
```python
from src.graphs.dfg import DFGBuilder

dfg_builder = DFGBuilder()
dfg = dfg_builder.build(parse_result, focus_lines=[10, 15])  # Chỉ build quanh focus
```

### Variable access types
- **PARAM**: parameter_declaration
- **DEF**: declaration, init_declarator, assignment_expression
- **USE**: RHS của assignment, return vars, call arguments

### ⚠️ Hạn chế
- Reaching definition chưa implement đầy đủ → def-use mang tính heuristic theo thứ tự dòng
- Bỏ qua identifier toàn chữ hoa (macro/constant) → có thể miss tín hiệu

---

## 4. Slicing

### Slice Types
```python
from src.slicing.slicer import CodeSlicer, SliceConfig, SliceType

# Backward: tìm statements ảnh hưởng tới criterion (phổ biến cho vuln)
config = SliceConfig(slice_type=SliceType.BACKWARD)

# Forward: tìm statements bị ảnh hưởng bởi criterion (hữu ích cho taint analysis)
config = SliceConfig(slice_type=SliceType.FORWARD)

# Both: union backward + forward
config = SliceConfig(slice_type=SliceType.BOTH)

# Window: ±k lines quanh criterion (fallback khi parse fail)
config = SliceConfig(slice_type=SliceType.WINDOW, window_size=15)
```

### Multi-slice (Backward + Forward combined)
```python
from src.slicing.multi_slicer import MultiCodeSlicer, MultiSliceConfig

config = MultiSliceConfig(
    backward_depth=5,
    backward_window=15,
    forward_depth=3,
    forward_window=10,
    max_combined_tokens=512,
    sep_token='[SEP]',
    include_control_deps=True,
    include_data_deps=True,
)
slicer = MultiCodeSlicer(config)
result = slicer.multi_slice(code, criterion_lines)
combined_code = result.combined_code  # backward [SEP] forward
```

### Slicing algorithm
1. Validate criterion_lines nằm trong [1..max_line]
2. Remove comments (optional, preserve newlines)
3. Build CFG/DFG nếu chưa có
4. Nếu parse/graph fail → **window_slice fallback**
5. Backward: BFS ngược qua DFG (data deps) + CFG (control deps)
6. Mở rộng lặp theo `max_depth`

### ⚠️ Vấn đề Line-number mismatch
Khi `remove_comments=True`, comment removal phải **giữ nguyên số newline** (thay comment bằng spaces nhưng giữ `\n`). Nếu không, criterion lines sẽ lệch.

---

## 5. Tokenization

### 3 loại tokenizer

#### A) HybridTokenizer (normalize identifiers)
```python
from src.tokenization.hybrid_tokenizer import HybridTokenizer

tokenizer = HybridTokenizer(
    preserve_dangerous_apis=True,
    preserve_keywords=True
)
tokens = tokenizer.tokenize(code)
```

**Normalization rules:**
- String → `STR`, Char → `CHAR`, Number → `NUM`, Float → `FLOAT`
- Dangerous API (được gọi) → giữ nguyên tên
- Defense API → giữ nguyên tên
- Function call khác → `FUNC`
- Identifier khác → `ID`

#### B) CanonicalTokenizer (semantic variable buckets)
```python
from src.tokenization.hybrid_tokenizer import CanonicalTokenizer

tokenizer = CanonicalTokenizer()
tokens = tokenizer.tokenize(code)
```

**Variable buckets:**
- `BUF_k`: buffer, dst, src, data, payload...
- `LEN_k`: len, size, count...
- `PTR_k`: ptr, pointer, head, tail, node...
- `IDX_k`: idx, index, pos, offset...
- `SENS_k`: password, token, secret, credential, auth, session...
- `PRIV_k`: admin, root, user, privilege, permission...
- `VAR_k`: other variables

#### C) PreserveIdentifierTokenizer
```python
from src.tokenization.preserve_tokenizer import PreserveIdentifierTokenizer

tokenizer = PreserveIdentifierTokenizer(config)
tokens, details, stats = tokenizer.tokenize_batch(codes, with_details=True)
```

### Dangerous APIs (luôn được preserve)
```python
DANGEROUS_APIS = {
    # Memory allocation
    'malloc', 'free', 'realloc', 'calloc', 'alloca',
    # String (unsafe)
    'strcpy', 'strcat', 'sprintf', 'gets', 'scanf', 'vsprintf',
    # Memory ops
    'memcpy', 'memmove', 'memset', 'bcopy',
    # Format string
    'printf', 'fprintf', 'snprintf', 'vprintf',
    # File I/O
    'fopen', 'fread', 'fwrite', 'open', 'read', 'write',
    # System
    'system', 'exec', 'popen', 'fork',
    ...
}
```

### Defense APIs (cũng được preserve)
```python
DEFENSE_APIS = {'strlen', 'sizeof', 'assert', 'check', 'validate', 'verify'}
DEFENSE_PATTERNS = ['safe_', 'check_', 'validate_', 'verify_', 'is_valid']
```

---

## 6. Vocabulary & Vectorization

### Build vocab từ train data only
```python
from src.tokenization.hybrid_tokenizer import build_hybrid_vocab, vectorize

vocab = build_hybrid_vocab(
    train_codes,
    min_freq=2,
    max_size=30000,
    use_canonical=True
)

# Dangerous APIs được forced vào vocab (không bị UNK)
```

### Special tokens
```python
SPECIAL_TOKENS = {
    'PAD': 0,   # Padding
    'UNK': 1,   # Unknown
    'BOS': 2,   # Begin of sequence
    'EOS': 3,   # End of sequence
    'SEP': 4,   # Separator (quan trọng cho multi-slice!)
}
```

### Vectorize
```python
input_ids, attention_mask = vectorize(tokens, vocab, max_len=512)
```

---

# TRAINING WORKFLOW

## Model Architecture

**Hybrid BiGRU + Attention + Optional Vuln Features MLP**

```
Input (input_ids, attention_mask)
    ↓
Embedding (embed_dim=64)
    ↓
BiGRU (hidden_dim=128, num_layers=1, bidirectional=True)
    ↓
Additive Attention pooling
    ↓
[Optional: Concat với MLP features từ vuln_features]
    ↓
Classifier MLP → Sigmoid
```

## Training Config
```python
# Default config
config = {
    'embed_dim': 64,
    'hidden_dim': 128,
    'num_layers': 1,
    'bidirectional': True,
    'embedding_dropout': 0.3,
    'classifier_dropout': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'epochs': 30,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'grad_clip': 1.0,
    'patience': 7,
    'use_amp': True,  # Mixed precision
}
```

## Loss & Optimization

- **Loss**: BCEWithLogitsLoss + class weights + label smoothing
- **Optimizer**: AdamW với weight_decay
- **Scheduler**: OneCycleLR hoặc ReduceLROnPlateau
- **AMP**: autocast + GradScaler cho Kaggle T4
- **Multi-GPU**: DataParallel

## Threshold Optimization
```python
# Tìm threshold tốt nhất trên validation để maximize F1
threshold_config = {
    'threshold_min': 0.3,
    'threshold_max': 0.7,
    'threshold_step': 0.01,
    'use_optimal_threshold': True,
}
```

---

# DEBUG COMMON ISSUES

## 1. Tree-sitter import error
**Triệu chứng:** `ImportError: Required packages missing...`

**Fix:**
```bash
pip install tree-sitter tree-sitter-c tree-sitter-cpp
```

## 2. Slicer luôn fallback window
**Triệu chứng:** Tỉ lệ window fallback cao, slice "kém thông minh"

**Nguyên nhân:**
- `parse()` return `None` khi exception
- CFG/DFG build fail (return None hoặc empty)

**Fix:**
- Log/đếm tỉ lệ fallback
- Tắt `remove_comments` để test
- Giảm `max_depth`, tắt `include_control_deps`

## 3. Line-number mismatch khi remove comments
**Triệu chứng:** Criterion lines không đúng vị trí, slice miss vùng vuln

**Fix:**
- Đảm bảo comment removal giữ nguyên số newline
- Hoặc slice trên original code, remove comment sau

## 4. Multiprocessing treo trên Kaggle
**Triệu chứng:** `ProcessPoolExecutor` treo hoặc lỗi pickle

**Fix:**
```python
PROCESS_CONFIG = {
    'n_jobs': 1,  # Tắt multiprocessing
}
```

## 5. UNK rate cao (>5%)
**Triệu chứng:** Nhiều token thành `UNK`, model học kém

**Fix:**
- Tăng `max_vocab_size` (30k → 50k)
- Giảm `min_freq` (2 → 1)
- Check dangerous APIs có trong vocab không
- Xem `vocab_debug.json` cho top UNK tokens

## 6. SEP token missing
**Triệu chứng:** Multi-slice không hoạt động đúng

**Fix:**
```python
if 'SEP' not in vocab:
    raise ValueError("SEP token missing!")
# SEP phải có ID=4
```

## 7. Training overfitting
**Triệu chứng:** Train F1 cao, Val F1 thấp

**Fix:**
- Giảm capacity: `hidden_dim`, `num_layers`
- Tăng dropout: 0.3 → 0.5
- Tăng `weight_decay`
- Bật `label_smoothing`
- Early stopping với `patience` thấp hơn

## 8. AMP instability (nan/inf)
**Triệu chứng:** Loss trở thành nan

**Fix:**
- Giữ `GradScaler` enabled
- Giảm `max_lr`
- Bật `grad_clip`
- Tạm tắt AMP: `use_amp=False`

## 9. Data leakage
**Triệu chứng:** Val/Test metrics quá cao, không reproducible

**Fix:**
```python
# Luôn remove cross-split duplicates
train_val_overlap = set(train_df['func']) & set(val_df['func'])
val_df = val_df[~val_df['func'].isin(train_val_overlap)]

# Remove internal duplicates
train_df = train_df.drop_duplicates(subset='func', keep='first')
```

---

# HYPERPARAMETER TUNING

## Slicing (ảnh hưởng input quality mạnh nhất)
| Param | Default | Mô tả |
|-------|---------|-------|
| `backward_depth` | 5 | Số hop lan truyền backward |
| `forward_depth` | 3 | Số hop lan truyền forward |
| `backward_window` | 15 | Window fallback cho backward |
| `forward_window` | 10 | Window fallback cho forward |
| `include_control_deps` | True | Bao gồm control dependencies |
| `include_data_deps` | True | Bao gồm data dependencies |
| `max_combined_tokens` | 512 | Max tokens sau combine |

## Tokenization
| Param | Default | Mô tả |
|-------|---------|-------|
| `min_freq` | 2 | Min frequency để vào vocab |
| `max_vocab_size` | 30000 | Max vocab size |
| `max_seq_length` | 512 | Max sequence length |
| `preserve_dangerous_apis` | True | Giữ nguyên tên dangerous APIs |

## Model
| Param | Default | Mô tả |
|-------|---------|-------|
| `embed_dim` | 64 | Embedding dimension |
| `hidden_dim` | 128 | GRU hidden dimension |
| `num_layers` | 1 | Số layers GRU |
| `embedding_dropout` | 0.3 | Dropout sau embedding |
| `classifier_dropout` | 0.3 | Dropout trong classifier |

## Training
| Param | Default | Mô tả |
|-------|---------|-------|
| `batch_size` | 128 | Batch size (Kaggle T4) |
| `learning_rate` | 1e-3 | Learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `label_smoothing` | 0.1 | Label smoothing |
| `grad_clip` | 1.0 | Gradient clipping |
| `patience` | 7 | Early stopping patience |

---

# BEST PRACTICES

1. **Chống data leakage**: Luôn dedup cross-split + internal dedup trước khi train

2. **Monitor fallback rate**: Nếu window fallback > 30%, cần cải thiện parse/graph

3. **Giữ nhất quán line mapping**: Slice trước, remove comment sau (hoặc đảm bảo comment removal preserve newlines)

4. **Ablation testing**: Chạy lần lượt window-only → backward-only → backward+control → both để xác định component nào gây issue

5. **Kaggle T4 optimization**:
   - AMP + batch_size lớn (128) để tận dụng VRAM
   - `num_workers` thấp (Kaggle 2 vCPU)
   - `n_jobs=1` cho slicing (avoid multiprocessing issues)

6. **Threshold reporting**: Luôn report cả threshold=0.5 và threshold optimized để tránh ảo tưởng metric

7. **Vocab validation**: Đảm bảo SEP, dangerous APIs, defense APIs đều trong vocab

---

# COMMANDS

```bash
# Check syntax
python3 -m py_compile devign_pipeline/notebooks/prepare_data.py

# Zip for Kaggle upload
zip -r devign_pipeline.zip devign_pipeline/

# Local run
export DEVIGN_DATA_DIR=/path/to/dataset
export DEVIGN_OUTPUT_DIR=/path/to/output
python3 devign_pipeline/notebooks/prepare_data.py
```

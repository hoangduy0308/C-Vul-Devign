# 🔍 Devign Vulnerability Scanner - CI/CD Integration Guide

AI-powered vulnerability detection for C/C++ code using BiGRU deep learning.

## 📦 Components

```
devign_pipeline/
├── devign_infer/          # Inference library
│   ├── __init__.py
│   ├── config.py          # Configuration classes
│   ├── detector.py        # Main VulnerabilityDetector class
│   └── sarif.py           # SARIF report generator
├── devign_scan.py         # CLI scanner
└── models/                # Model files (after training)
    ├── best_model.pt
    └── vocab.json

.github/workflows/
└── devign-scan.yml        # GitHub Actions workflow
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy tqdm
```

### 2. Prepare Model Files

After training, copy model files to `models/`:
```bash
mkdir -p models
cp output/models/best_model.pt models/
cp output/vocab.json models/
```

### 3. Run Scanner

```bash
cd devign_pipeline

# Scan a single file
python devign_scan.py scan vulnerable.c

# Scan a directory
python devign_scan.py scan src/ --recursive

# Output SARIF for GitHub
python devign_scan.py scan src/ -o results.sarif -f sarif
```

## 🔧 CLI Usage

### Basic Commands

```bash
# Scan files/directory
python devign_scan.py scan <path> [options]

# Scan git diff (for CI)
python devign_scan.py scan-diff --base origin/main [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--model, -m` | Path to model checkpoint |
| `--vocab, -v` | Path to vocabulary file |
| `--threshold, -t` | Probability threshold (default: 0.5) |
| `--format, -f` | Output format: text, json, sarif |
| `--output, -o` | Output file path |
| `--fail-on-findings` | Exit code 1 if vulnerabilities found |
| `--quiet, -q` | Suppress progress output |

### Examples

```bash
# Scan with custom threshold
python devign_scan.py scan src/ --threshold 0.7

# Generate JSON report
python devign_scan.py scan src/ -f json -o report.json

# CI mode: fail if vulnerabilities found
python devign_scan.py scan src/ --fail-on-findings -f sarif -o results.sarif
```

## 🔗 GitHub Actions Integration

### Automatic Setup

The workflow at `.github/workflows/devign-scan.yml` automatically:

1. **On Pull Requests**: Scans only changed C/C++ files
2. **On Push to main**: Scans changed files since last commit
3. **Uploads SARIF**: Results appear in Security → Code Scanning

### Required Setup

#### 1. Enable Code Scanning

Go to: `Settings → Security → Code scanning → Setup → Advanced`

#### 2. Upload Model Files

**Option A: GitHub Releases**
```bash
# Create release with model files
gh release create v1.0 models/best_model.pt models/vocab.json
```

Then update workflow to download from release.

**Option B: Git LFS**
```bash
git lfs install
git lfs track "*.pt"
git add models/
git commit -m "Add model files"
```

**Option C: External Storage**
Upload to S3/GCS and configure download URL as repository secret.

#### 3. Repository Secrets (if using external storage)

- `MODEL_URL`: URL to download best_model.pt
- `VOCAB_URL`: URL to download vocab.json

### Viewing Results

After a scan runs:

1. Go to **Security** tab
2. Click **Code scanning alerts**
3. View vulnerability details and affected lines

### Customizing Workflow

Edit `.github/workflows/devign-scan.yml`:

```yaml
# Change threshold
--threshold 0.7

# Block PRs with vulnerabilities (uncomment in workflow)
exit 1
```

## 🐳 Docker Usage

### Build Image

```bash
docker build -t devign-scanner:latest .
```

### Run Scan

```bash
# Mount code and models
docker run -v /path/to/code:/code \
           -v /path/to/models:/app/models \
           devign-scanner:latest scan /code

# Generate SARIF
docker run -v $(pwd):/code \
           -v ./models:/app/models \
           devign-scanner:latest scan /code -f sarif -o /code/results.sarif
```

## 📊 Output Formats

### Text (Human-readable)

```
============================================================
DEVIGN VULNERABILITY SCAN RESULTS
============================================================
Files scanned: 10
Vulnerabilities found: 2
------------------------------------------------------------
🟠 src/buffer.c
   Risk: HIGH (87.3%)
   APIs: strcpy, memcpy, malloc
🟡 src/parser.c
   Risk: MEDIUM (62.1%)
   APIs: sprintf, atoi
============================================================
⚠️  Found 2 potential vulnerabilities!
```

### JSON

```json
{
  "summary": {
    "files_scanned": 10,
    "vulnerabilities_found": 2,
    "errors": 0
  },
  "results": [
    {
      "file": "src/buffer.c",
      "vulnerable": true,
      "probability": 0.873,
      "risk_level": "HIGH",
      "dangerous_apis": ["strcpy", "memcpy", "malloc"]
    }
  ]
}
```

### SARIF (GitHub Code Scanning)

Standard SARIF 2.1.0 format with:
- Rule definitions for vulnerability types
- Location information (file, line)
- Severity levels (error, warning)
- Fingerprints for deduplication

## 🔒 Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| CRITICAL | ≥ 90% | Immediate review required |
| HIGH | ≥ 75% | Priority fix |
| MEDIUM | ≥ 50% | Schedule fix |
| LOW | ≥ 25% | Monitor |
| NONE | < 25% | No action needed |

## ⚙️ Python API

```python
from devign_infer import VulnerabilityDetector

# Initialize detector
detector = VulnerabilityDetector(
    model_path="models/best_model.pt",
    vocab_path="models/vocab.json",
    threshold=0.5
)

# Analyze code
result = detector.analyze("""
int vulnerable(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // Buffer overflow!
    return 0;
}
""")

print(f"Vulnerable: {result.vulnerable}")
print(f"Probability: {result.probability:.1%}")
print(f"Risk Level: {result.risk_level}")
print(f"Dangerous APIs: {result.details['dangerous_apis_found']}")
```

## 🛠️ Troubleshooting

### Model not found
```
Error: No such file or directory: 'models/best_model.pt'
```
→ Ensure model files are in the correct path or specify with `--model` flag.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
→ Use `--device cpu` flag or reduce batch size.

### No C files found
```
No C/C++ files found in src/
```
→ Check file extensions (.c, .h, .cpp, .hpp)

## 📝 License

MIT License - See LICENSE file for details.

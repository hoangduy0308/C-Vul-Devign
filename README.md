# C-Vul-Devign

End-to-end C/C++ vulnerability detection project covering dataset preprocessing, model training, inference, SARIF reporting, Docker packaging, and GitHub Action integration.

## Overview

This repository combines machine learning and program analysis for source-code security scanning. It is structured as a working research and engineering codebase rather than a single demo script.

Core areas:

- preprocessing pipeline for Devign-style parquet datasets
- C/C++ vulnerability scanner with JSON and SARIF output
- BiGRU-based training and inference code
- vulnerability-pattern rules and graph-aware slicing
- packaging for CI/CD and containerized use

## Main Components

### Preprocessing pipeline

- Entry point: [devign_pipeline/cli.py](devign_pipeline/cli.py)
- Core implementation: [devign_pipeline/src/pipeline/preprocess.py](devign_pipeline/src/pipeline/preprocess.py)

Pipeline stages:

1. `load`
2. `vuln_features`
3. `ast`
4. `cfg`
5. `dfg`
6. `slice`
7. `tokenize`
8. `normalize`
9. `vocab`
10. `vectorize`

### Scanner

- Entry point: [devign_pipeline/devign_scan.py](devign_pipeline/devign_scan.py)
- Output formats: `text`, `json`, `sarif`
- Scan modes: full path scan and Git diff scan

Supported file types:

- `.c`
- `.h`
- `.cpp`
- `.hpp`
- `.cc`
- `.cxx`
- `.hxx`

### Training and inference

- Training code: [devign_pipeline/src/training](devign_pipeline/src/training)
- Inference modules: [devign_pipeline/devign_infer](devign_pipeline/devign_infer)
- Extended API inference: [devign_pipeline/api/inference.py](devign_pipeline/api/inference.py)
- Vulnerability rules: [devign_pipeline/config/vuln_patterns.yaml](devign_pipeline/config/vuln_patterns.yaml)

### Delivery artifacts

- GitHub Action: [action.yml](action.yml)
- Docker image: [Dockerfile](Dockerfile)
- Release packaging: [build_scanner.py](build_scanner.py)

## Repository Structure

```text
.
|-- action.yml
|-- Dockerfile
|-- build_scanner.py
|-- create_zip.py
|-- c_vuln_scanner/
|-- devign_pipeline/
|   |-- api/
|   |-- cli/
|   |-- config/
|   |-- devign_infer/
|   |-- src/
|   |-- tests/
|   |-- cli.py
|   `-- devign_scan.py
|-- Dataset/
|   |-- devign/
|   |-- devign_final/
|   `-- c_vuln_scanner/
|-- models/
|-- docs/
`-- Codemau/
```

## Data and Model Artifacts

Expected dataset split format:

- `train-*.parquet`
- `validation-*.parquet`
- `test-*.parquet`

Model artifacts in the current workspace include:

- `best_v2_seed42.pt`
- `best_v2_seed1042.pt`
- `best_v2_seed2042.pt`
- `models/config.json`
- `models/vocab.json`

Reference ensemble metrics from [c_vuln_scanner/ensemble_config.json](c_vuln_scanner/ensemble_config.json):

- optimal threshold: `0.65`
- test F1: `0.7727`
- test precision: `0.8022`
- test recall: `0.7452`
- test AUC: `0.8783`

## Quick Start

Install minimal inference dependencies:

```bash
pip install -r requirements-inference.txt
```

Show preprocessing pipeline info:

```bash
python devign_pipeline/cli.py info
```

Run preprocessing:

```bash
python devign_pipeline/cli.py run \
  --data-dir Dataset/devign \
  --output-dir output/processed \
  --checkpoint-dir output/checkpoints
```

Show scanner help:

```bash
python devign_pipeline/devign_scan.py --help
```

Generate SARIF:

```bash
python devign_pipeline/devign_scan.py scan src/ \
  --format sarif \
  --output results.sarif
```

Run tests:

```bash
pytest devign_pipeline/tests/test_optimized_tokenizer.py -q
```

## Tech Stack

- Python
- PyTorch
- FastAPI
- tree-sitter
- NetworkX
- pandas
- GitHub Actions
- Docker
- SARIF

## Current Status

Verified in the current workspace:

- `python devign_pipeline/devign_scan.py --help` works
- `python devign_pipeline/cli.py info` works
- `pytest devign_pipeline/tests/test_optimized_tokenizer.py -q` passes with `18 passed`

Known limitations:

- the root FastAPI layer is not fully import-clean
- some scanner artifacts are currently out of sync with the latest vocab/config artifacts
- the repository includes both active development code and packaged snapshots

## References

Background material is listed in [docs/references.md](docs/references.md).

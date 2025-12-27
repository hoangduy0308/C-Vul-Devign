#!/usr/bin/env python3
"""
Build devign-scanner.zip for GitHub Releases

This script creates a self-contained scanner package that can be used
as a GitHub Action in any C/C++ repository.

Usage:
    python build_scanner.py
    
Output:
    devign-scanner.zip - Ready to upload to GitHub Releases
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_NAME = "devign-scanner.zip"
BUILD_DIR = SCRIPT_DIR / "build" / "scanner"

# Files/folders to include
INCLUDE_ITEMS = [
    # Main scanner script
    ("devign_pipeline/devign_scan.py", "devign_scan.py"),
    
    # Inference module
    ("devign_pipeline/devign_infer", "devign_infer"),
    
    # Source modules (needed for tokenization, etc.)
    ("devign_pipeline/src/__init__.py", "src/__init__.py"),
    ("devign_pipeline/src/tokenization", "src/tokenization"),
    ("devign_pipeline/src/ast", "src/ast"),
    ("devign_pipeline/src/slicing", "src/slicing"),
    ("devign_pipeline/src/vuln", "src/vuln"),
    ("devign_pipeline/src/utils", "src/utils"),
    ("devign_pipeline/src/training", "src/training"),
    
    # Models and config
    ("models/best_v2_seed42.pt", "models/best_v2_seed42.pt"),
    ("models/best_v2_seed1042.pt", "models/best_v2_seed1042.pt"),
    ("models/best_v2_seed2042.pt", "models/best_v2_seed2042.pt"),
    ("models/vocab.json", "models/vocab.json"),
    ("models/config.json", "models/config.json"),
    ("models/feature_stats.json", "models/feature_stats.json"),
]

# Patterns to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".git",
    ".pytest_cache",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
]


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    name = path.name
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
    return False


def copy_item(src: Path, dst: Path):
    """Copy file or directory, excluding unwanted files."""
    if should_exclude(src):
        return
    
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  + {dst.relative_to(BUILD_DIR)}")
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if not should_exclude(item):
                copy_item(item, dst / item.name)


def create_requirements_txt():
    """Create minimal requirements.txt for scanner."""
    requirements = """\
# Devign Scanner Dependencies
# Install with: pip install -r requirements.txt

torch>=2.0.0
numpy>=1.20.0
tqdm>=4.60.0
"""
    req_path = BUILD_DIR / "requirements.txt"
    req_path.write_text(requirements)
    print(f"  + requirements.txt")


def create_readme():
    """Create README for the scanner package."""
    readme = """\
# Devign Vulnerability Scanner

AI-powered C/C++ vulnerability detection using BiGRU deep learning.

## Quick Start

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tqdm

# Scan a file
python devign_scan.py scan file.c

# Scan a directory
python devign_scan.py scan src/ --recursive

# Output SARIF for GitHub Code Scanning
python devign_scan.py scan src/ --output results.sarif --format sarif
```

## Usage in GitHub Actions

```yaml
- uses: hoangduy0308/C-Vul-Devign@main
  with:
    threshold: 0.5
    scan-mode: diff
```

## Options

- `--threshold`: Vulnerability probability threshold (default: 0.5)
- `--format`: Output format - text, json, sarif (default: text)
- `--output`: Save results to file
- `--recursive`: Scan directories recursively
- `--fail-on-findings`: Exit with code 1 if vulnerabilities found

## License

MIT License
"""
    readme_path = BUILD_DIR / "README.md"
    readme_path.write_text(readme)
    print(f"  + README.md")


def build_scanner():
    """Build the scanner package."""
    print("=" * 60)
    print("Building Devign Scanner Package")
    print("=" * 60)
    
    # Clean build directory
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)
    
    print(f"\nBuild directory: {BUILD_DIR}")
    print(f"\nCopying files...")
    
    # Copy all items
    missing_files = []
    for src_rel, dst_rel in INCLUDE_ITEMS:
        src = SCRIPT_DIR / src_rel
        dst = BUILD_DIR / dst_rel
        
        if not src.exists():
            missing_files.append(src_rel)
            print(f"  ! MISSING: {src_rel}")
            continue
        
        copy_item(src, dst)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files missing!")
        print("   The scanner may not work correctly.")
        print("   Missing files:")
        for f in missing_files:
            print(f"     - {f}")
    
    # Create additional files
    print(f"\nGenerating files...")
    create_requirements_txt()
    create_readme()
    
    # Create zip file
    print(f"\nCreating zip archive...")
    output_path = SCRIPT_DIR / OUTPUT_NAME
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in BUILD_DIR.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(BUILD_DIR)
                zf.write(file_path, arcname)
    
    # Calculate size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n" + "=" * 60)
    print(f"[OK] Build complete!")
    print(f"=" * 60)
    print(f"Output: {output_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"\nNext steps:")
    print(f"  1. Create a GitHub Release")
    print(f"  2. Upload {OUTPUT_NAME} as release asset")
    print(f"  3. Use in other repos with:")
    print(f"     - uses: hoangduy0308/C-Vul-Devign@main")
    
    return 0


def main():
    try:
        return build_scanner()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
# =============================================================================
# Devign Scanner - Build & Publish Release Script
#
# Usage:
#   ./scripts/publish_release.sh v1.0.0
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated
#   - Model files in output/models/
#   - Vocabulary file exists
# =============================================================================

set -e

VERSION=${1:-"v1.0.0"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"
RELEASE_DIR="$BUILD_DIR/release"

echo "=========================================="
echo "Devign Scanner Release Builder"
echo "Version: $VERSION"
echo "=========================================="

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$RELEASE_DIR"

echo ""
echo "[1/5] Copying scanner files..."
mkdir -p "$RELEASE_DIR/devign_infer"
mkdir -p "$RELEASE_DIR/models"

# Copy inference library
cp -r "$ROOT_DIR/devign_pipeline/devign_infer/"* "$RELEASE_DIR/devign_infer/"

# Copy CLI scanner
cp "$ROOT_DIR/devign_pipeline/devign_scan.py" "$RELEASE_DIR/"

# Copy requirements
cp "$ROOT_DIR/requirements-inference.txt" "$RELEASE_DIR/requirements.txt"

echo "[2/5] Copying model files..."
# Check for model files
if [ -f "$ROOT_DIR/output/models/best_model.pt" ]; then
    cp "$ROOT_DIR/output/models/best_model.pt" "$RELEASE_DIR/models/"
    echo "  ✓ best_model.pt"
else
    echo "  ⚠ Warning: best_model.pt not found in output/models/"
fi

# Check for vocab file
VOCAB_FOUND=false
for vocab_path in "$ROOT_DIR/output/vocab.json" "$ROOT_DIR/models/vocab.json" "$ROOT_DIR/devign_pipeline/vocab.json"; do
    if [ -f "$vocab_path" ]; then
        cp "$vocab_path" "$RELEASE_DIR/models/vocab.json"
        echo "  ✓ vocab.json (from $vocab_path)"
        VOCAB_FOUND=true
        break
    fi
done

if [ "$VOCAB_FOUND" = false ]; then
    echo "  ⚠ Warning: vocab.json not found"
fi

# Check for config
if [ -f "$ROOT_DIR/output/config.json" ]; then
    cp "$ROOT_DIR/output/config.json" "$RELEASE_DIR/models/"
    echo "  ✓ config.json"
fi

echo "[3/5] Creating release package..."
cd "$BUILD_DIR"

# Create scanner-only package (without models, for fast download)
zip -r devign-scanner-code.zip release/devign_infer release/devign_scan.py release/requirements.txt

# Create full package with models
zip -r devign-scanner-full.zip release/

# Create models-only package
if [ -f "release/models/best_model.pt" ]; then
    zip -r devign-models.zip release/models/
fi

echo "  ✓ devign-scanner-code.zip (code only)"
echo "  ✓ devign-scanner-full.zip (with models)"
echo "  ✓ devign-models.zip (models only)"

echo ""
echo "[4/5] Package sizes:"
ls -lh "$BUILD_DIR"/*.zip

echo ""
echo "[5/5] Creating GitHub Release..."

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "⚠ GitHub CLI (gh) not installed. Skipping release upload."
    echo ""
    echo "Manual upload instructions:"
    echo "  1. Go to https://github.com/hoangduy0308/C-Vul-Devign/releases/new"
    echo "  2. Create tag: $VERSION"
    echo "  3. Upload files from: $BUILD_DIR/"
    exit 0
fi

# Create release
cd "$ROOT_DIR"

RELEASE_NOTES="## Devign Vulnerability Scanner $VERSION

### Downloads
- **devign-scanner-code.zip**: Scanner code only (lightweight)
- **devign-scanner-full.zip**: Complete package with trained models
- **devign-models.zip**: Model files only

### Quick Start
\`\`\`bash
# Download and extract
curl -L https://github.com/hoangduy0308/C-Vul-Devign/releases/download/$VERSION/devign-scanner-full.zip -o scanner.zip
unzip scanner.zip

# Scan your code
pip install torch numpy tqdm
python release/devign_scan.py scan /path/to/code
\`\`\`

### GitHub Actions
\`\`\`yaml
- uses: hoangduy0308/C-Vul-Devign@$VERSION
  with:
    threshold: '0.5'
\`\`\`
"

echo "$RELEASE_NOTES" > "$BUILD_DIR/RELEASE_NOTES.md"

# Check if release exists
if gh release view "$VERSION" &> /dev/null; then
    echo "Release $VERSION already exists. Updating..."
    gh release upload "$VERSION" \
        "$BUILD_DIR/devign-scanner-code.zip" \
        "$BUILD_DIR/devign-scanner-full.zip" \
        "$BUILD_DIR/devign-models.zip" \
        --clobber
else
    echo "Creating new release $VERSION..."
    gh release create "$VERSION" \
        "$BUILD_DIR/devign-scanner-code.zip" \
        "$BUILD_DIR/devign-scanner-full.zip" \
        "$BUILD_DIR/devign-models.zip" \
        --title "Devign Scanner $VERSION" \
        --notes-file "$BUILD_DIR/RELEASE_NOTES.md"
fi

echo ""
echo "=========================================="
echo "✅ Release $VERSION published successfully!"
echo ""
echo "View at: https://github.com/hoangduy0308/C-Vul-Devign/releases/tag/$VERSION"
echo "=========================================="

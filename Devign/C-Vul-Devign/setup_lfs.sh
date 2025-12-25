# Git LFS Configuration

## Setup Git LFS for model files

# Install Git LFS (if not already installed)
# Ubuntu/Debian: sudo apt install git-lfs
# macOS: brew install git-lfs

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "models/*.pt"
git lfs track "models/*.pth"
git lfs track "*.pt"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes

# Verify tracking
git lfs track

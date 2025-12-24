import zipfile
import os

src = 'f:/Work/C Vul Devign/devign_pipeline'
dst = 'f:/Work/C Vul Devign/devign_pipeline.zip'

# Directories and file patterns to exclude
EXCLUDE_DIRS = {'__pycache__', '.git', '.ipynb_checkpoints', 'venv', 'checkpoints'}
EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.DS_Store'}
EXCLUDE_FILES = {'prepare_data.py', '03_preprocessing_v2.py'}  # Deleted old files

# Remove old zip if exists
if os.path.exists(dst):
    os.remove(dst)
    print(f'Removed old: {dst}')

count = 0
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            # Skip excluded extensions
            if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                continue
            # Skip excluded files
            if file in EXCLUDE_FILES:
                continue
                
            full_path = os.path.join(root, file)
            # Create archive name with forward slashes (for Linux/Kaggle compatibility)
            arc_name = os.path.relpath(full_path, src).replace('\\', '/')
            zf.write(full_path, arc_name)
            print(f'  {arc_name}')
            count += 1

size_kb = os.path.getsize(dst) / 1024
print(f'\n=== Created: devign_pipeline.zip ===')
print(f'Files: {count}')
print(f'Size: {size_kb:.1f} KB')

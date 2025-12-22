"""Create zip file with forward slashes for Kaggle compatibility"""
import zipfile
import os

source_dir = 'F:/Work/C Vul Devign/devign_pipeline'
zip_path = 'F:/Work/C Vul Devign/devign_pipeline.zip'

EXCLUDE_DIRS = {'__pycache__', '.git', 'node_modules', '.venv', 'venv'}
EXCLUDE_EXTS = {'.pyc', '.pyo', '.pyd', '.so', '.dll'}

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            # Skip excluded extensions
            if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
                
            file_path = os.path.join(root, file)
            # Use forward slashes for arcname (Linux/Kaggle compatibility)
            arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
            arcname = arcname.replace('\\', '/')
            zipf.write(file_path, arcname)
            print(f"Added: {arcname}")

print(f"\nCreated: {zip_path}")
print(f"Total files: {len(zipf.namelist())}")

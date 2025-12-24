import zipfile
import os

src = 'f:/Work/C Vul Devign/codebert'
dst = 'f:/Work/C Vul Devign/codebert.zip'

EXCLUDE_DIRS = {'__pycache__', '.git', '.ipynb_checkpoints', 'venv'}
EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.DS_Store'}

if os.path.exists(dst):
    os.remove(dst)
    print(f'Removed old: {dst}')

count = 0
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                continue
                
            full_path = os.path.join(root, file)
            # Forward slashes for Linux/Kaggle
            arc_name = os.path.relpath(full_path, src).replace('\\', '/')
            info = zipfile.ZipInfo(arc_name)
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(full_path, 'rb') as f:
                zf.writestr(info, f.read())
            print(f'  {arc_name}')
            count += 1

size_kb = os.path.getsize(dst) / 1024
print(f'\n=== Created: codebert.zip ===')
print(f'Files: {count}')
print(f'Size: {size_kb:.1f} KB')

import zipfile
import os
from pathlib import Path

source_dir = Path('F:/Work/C Vul Devign/devign_pipeline')
zip_path = Path('F:/Work/C Vul Devign/devign_pipeline.zip')

# Remove old zip
if zip_path.exists():
    zip_path.unlink()
    print("Removed old zip")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file_path in source_dir.rglob('*'):
        if file_path.is_file():
            # Include devign_pipeline/ prefix with forward slashes
            arcname = 'devign_pipeline/' + file_path.relative_to(source_dir).as_posix()
            zf.write(file_path, arcname)
            
print(f'Created: devign_pipeline.zip')

# Verify
with zipfile.ZipFile(zip_path, 'r') as zf:
    print(f'Total files: {len(zf.namelist())}')
    print(f'Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB')
    print('Sample paths:')
    for n in zf.namelist()[:8]:
        print(f'  {n}')

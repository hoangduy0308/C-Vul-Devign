"""Create zip file with forward slashes for cross-platform compatibility."""
import zipfile
import os

zip_path = 'devign_pipeline.zip'
source_dir = 'devign_pipeline'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(source_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.pyc'):
                continue
            file_path = os.path.join(root, file)
            # Use forward slashes for cross-platform compatibility
            arcname = file_path.replace(os.sep, '/')
            zf.write(file_path, arcname)
            
print('Done! Files in zip:')
with zipfile.ZipFile(zip_path, 'r') as zf:
    for name in zf.namelist()[:15]:
        print(f'  {name}')
    if len(zf.namelist()) > 15:
        print(f'  ... and {len(zf.namelist()) - 15} more')
    print(f'\nTotal: {len(zf.namelist())} files')

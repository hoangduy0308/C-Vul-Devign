import zipfile
import os

src = 'f:/Work/C Vul Devign/devign_pipeline'
dst = 'f:/Work/C Vul Devign/devign_pipeline.zip'

# Remove old zip if exists
if os.path.exists(dst):
    os.remove(dst)

with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(src):
        for file in files:
            full_path = os.path.join(root, file)
            # Create archive name with forward slashes
            arc_name = os.path.relpath(full_path, src).replace('\\', '/')
            zf.write(full_path, arc_name)
            print(f'Added: {arc_name}')

print('Done!')

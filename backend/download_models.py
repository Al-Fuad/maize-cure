"""Download release weights and verify SHA-256 before making them available."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

root = Path(__file__).parent / 'models'
manifest = json.loads((root / 'manifest.json').read_text())
repo = os.environ.get('MODEL_REPOSITORY', 'Al-Fuad/maize-cure')
base = f'https://github.com/{repo}/releases/download/{manifest["release_tag"]}'
for asset in manifest['files']:
    target = root / asset['filename']
    if target.exists():
        with target.open('rb') as existing:
            valid = hashlib.file_digest(existing, 'sha256').hexdigest() == asset['sha256']
        if valid:
            print(f'Verified {target.name}')
            continue
    temporary = target.with_suffix('.download')
    try:
        with urllib.request.urlopen(f'{base}/{target.name}', timeout=120) as response, temporary.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        with temporary.open('rb') as downloaded:
            checksum = hashlib.file_digest(downloaded, 'sha256').hexdigest()
        if checksum != asset['sha256']:
            raise RuntimeError(f'Checksum mismatch: {target.name}')
        temporary.replace(target)
        print(f'Installed {target.name}')
    finally:
        temporary.unlink(missing_ok=True)

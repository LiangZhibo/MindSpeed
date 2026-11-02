import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source-manifest.json').read_text())
summary = {}
for arm in manifest:
    source = root / 'source' / arm
    for name, expected in manifest[arm]['files'].items():
        path = source / name
        if expected['type'] == 'symlink':
            assert path.is_symlink() and str(path.readlink()) == expected['target'], str(path)
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], str(path)
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() or p.is_symlink()}
    assert actual == set(manifest[arm]['files']), (arm, actual ^ set(manifest[arm]['files']))
    summary[arm] = dict(commit=manifest[arm]['commit'], files=len(actual))
print(json.dumps(dict(status='verified', sources=summary)), flush=True)

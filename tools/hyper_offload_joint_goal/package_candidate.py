"""Create an immutable source snapshot for independent candidate validation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

root = Path(__file__).resolve().parents[2]
tag = sys.argv[1]
assert tag.isalnum(), 'use an alphanumeric immutable revision name'
out = root / 'tools/hyper_offload_joint_goal' / tag
out.mkdir(exist_ok=False)
base = root / 'tools/hyper_offload_joint_goal/baseline/host-goal-selective-identity.tar.gz'
with tempfile.TemporaryDirectory(prefix='joint-source-') as temp:
    source = Path(temp)
    with tarfile.open(base) as tf:
        tf.extractall(source, filter='data')
    for folder in ('mindspeed/core/memory/hyper_offload',
                   'tests_extend/unit_tests/mindspeed/core/memory/hyper_offload'):
        for p in (root / folder).rglob('*.py'):
            target = source / p.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    entry = 'examples/qwen25_32b_fsdp2_a2/native_fsdp2/run_qwen25_native_mindspeed_fsdp2.py'
    shutil.copy2(root / entry, source / entry)
    for p in (root / 'tools/hyper_offload_joint_goal').glob('*.py'):
        target = source / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source.rglob('*') if p.is_file() and not p.is_symlink()}
    archive = out / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for p in sorted(source.rglob('*')):
            if p.is_file() or p.is_symlink():
                tf.add(p, arcname=str(p.relative_to(source)), recursive=False)
    manifest = dict(tag=tag, base='61d0ea932aa2242be3f90a85cf08c0b7f8a3fc6e',
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=files)
    (out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(directory=str(out), files=len(files), archive_sha256=manifest['archive_sha256'])))

"""Repackage verified source bytes from an existing archive plus a small delta.

Only upload transport is reused. The new tag still requires its own CPU/NPU
qualification. Every output file must match the complete frozen target map.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

base_tag, tag = sys.argv[1:3]
assert base_tag.isalnum() and tag.isalnum() and base_tag != tag
prefix = '/tmp/liangzhibo-joint-'
request = json.loads(Path(prefix + tag + '-package-request.json').read_text())
assert request['tag'] == tag
base = Path(prefix + base_tag + '.tar.gz')
delta = Path(prefix + tag + '-delta.tar.gz')
assert hashlib.sha256(base.read_bytes()).hexdigest() == request['base_archive_sha256']
assert hashlib.sha256(delta.read_bytes()).hexdigest() == request['delta_sha256']
root = Path('/tmp/liangzhibo-joint-package-' + tag)
kit, manifest_path = Path(prefix + tag + '.tar.gz'), Path(prefix + tag + '-manifest.json')
assert not root.exists() and not kit.exists() and not manifest_path.exists()
root.mkdir()
expected = request['target_files']


def install(archive, only_changed=False):
    names = set()
    with tarfile.open(archive) as source:
        for member in source:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            if member.name not in expected:
                assert member.name in request['removed'] and not only_changed
                continue
            assert member.name not in names
            names.add(member.name)
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.extractfile(member).read())
            os.chmod(target, member.mode & 0o777)
    if only_changed:
        assert names == set(request['changed'])


install(base)
install(delta, True)
assert {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()} == set(expected)
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
temporary = kit.with_suffix('.building')
with tarfile.open(temporary, 'w:gz') as output:
    for name in sorted(expected):
        output.add(root / name, arcname=name, recursive=False)
with tarfile.open(temporary) as output:
    assert {member.name for member in output.getmembers()} == set(expected)
    for member in output:
        assert hashlib.sha256(output.extractfile(member).read()).hexdigest() == expected[member.name]
temporary.replace(kit)
manifest = dict(tag=tag, base='61d0ea932aa2242be3f90a85cf08c0b7f8a3fc6e',
                archive_sha256=hashlib.sha256(kit.read_bytes()).hexdigest(), files=expected,
                equivalent_source_archive=request['equivalent_source_archive'],
                transport_base_archive=request['base_archive_sha256'], delta_sha256=request['delta_sha256'])
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}), flush=True)

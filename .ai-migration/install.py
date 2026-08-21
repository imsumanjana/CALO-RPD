#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PARTS = tuple(f'direct-part-{i:03d}' for i in range(13))
B64_SHA256 = 'cc09dff738a3a1dd5c9edf8cafc2c5b52299de524befd25a8f8fec9337dc4560'
TAR_SHA256 = '51dff026691e2984583d2d7bbcfb76b082e7ca731e6beb3ca81df08ce5c0b384'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(root: Path, *args: str) -> None:
    print('+', ' '.join(args))
    proc = subprocess.run(args, cwd=root)
    if proc.returncode:
        raise SystemExit(proc.returncode)


def safe_extract(tf: tarfile.TarFile, root: Path) -> None:
    root_resolved = root.resolve()
    for member in tf.getmembers():
        target = (root / member.name).resolve()
        if root_resolved != target and root_resolved not in target.parents:
            raise SystemExit(f'Unsafe archive member: {member.name}')
    tf.extractall(root)


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    if not (root / '.git').exists():
        print('Run this installer from a normal CALO-RPD Git checkout.', file=sys.stderr)
        return 2

    chunks: list[bytes] = []
    for name in PARTS:
        path = here / name
        if not path.is_file():
            print(f'Missing verified migration chunk: {path}', file=sys.stderr)
            return 2
        chunks.append(path.read_bytes())

    encoded = b''.join(b''.join(chunks).split())
    actual_b64 = sha256(encoded)
    if actual_b64 != B64_SHA256:
        print(
            'Verified migration transport checksum mismatch. Refusing to decode.\n'
            f'  expected base64 sha256: {B64_SHA256}\n'
            f'  actual base64 sha256:   {actual_b64}',
            file=sys.stderr,
        )
        return 2

    try:
        archive = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        print(f'Invalid verified base64 migration bundle: {exc}', file=sys.stderr)
        return 2
    actual_tar = sha256(archive)
    if actual_tar != TAR_SHA256:
        print(
            'Verified migration archive checksum mismatch. Refusing to extract.\n'
            f'  expected archive sha256: {TAR_SHA256}\n'
            f'  actual archive sha256:   {actual_tar}',
            file=sys.stderr,
        )
        return 2

    print(f'Migration transport verified: {actual_b64}')
    print(f'Migration archive verified:   {actual_tar}')

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp.write(archive)
        archive_path = Path(tmp.name)
    try:
        with tarfile.open(archive_path, 'r:gz') as tf:
            safe_extract(tf, root)
    finally:
        archive_path.unlink(missing_ok=True)

    # Remove transport before indexing so migration machinery never appears in
    # repository intelligence. The resulting deletions are expected working-tree changes.
    shutil.rmtree(here)

    gitignore = root / '.gitignore'
    current = gitignore.read_text(encoding='utf-8') if gitignore.exists() else ''
    additions = [entry for entry in ('.ai-cache/', '.ai-tmp/') if entry not in current.splitlines()]
    if additions:
        with gitignore.open('a', encoding='utf-8', newline='\n') as handle:
            if current and not current.endswith('\n'):
                handle.write('\n')
            handle.write('\n'.join(additions) + '\n')

    run(root, sys.executable, 'scripts/ai-agent-guard.py', '--repair', '--root', '.', '--canonical', 'AGENTS.md')
    run(root, sys.executable, 'scripts/ai-agent-guard.py', '--install-hook', '--root', '.')
    run(root, sys.executable, 'scripts/ai-index', 'init')
    run(root, sys.executable, 'scripts/ai-index', 'check')

    print('\nRepository intelligence v2 is installed and regenerated in the working tree.')
    print('The verified transport directory was removed before indexing.')
    print('No scientific workload or pytest suite was run by this installer.')
    print('Next run: powershell -ExecutionPolicy Bypass -File .\\validation\\validate_ai_repository_intelligence_v2.ps1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

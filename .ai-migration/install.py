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

PARTS = (
    'calo-ai-v2-part-00',
    'calo-ai-v2-part-01',
    'calo-ai-v2-part-02a',
    'calo-ai-v2-part-02b',
    'calo-ai-v2-part-02c',
    'calo-ai-v2-part-03',
)
B64_SHA256 = 'd2dbe063035935b11152b408ed704a397399ced643ea8213a5934f60ce63f681'
TAR_SHA256 = '3e20be96f8b7e1baedba4101303724f2f6b6ab86eecea6d0de64302be1194deb'


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

    chunks = []
    for name in PARTS:
        path = here / name
        if not path.is_file():
            print(f'Missing migration chunk: {path}', file=sys.stderr)
            return 2
        chunks.append(path.read_bytes())
    encoded = b''.join(chunks)
    actual_b64 = sha256(encoded)
    if actual_b64 != B64_SHA256:
        print(f'Migration bundle checksum mismatch: {actual_b64}', file=sys.stderr)
        return 2

    try:
        archive = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        print(f'Invalid base64 migration bundle: {exc}', file=sys.stderr)
        return 2
    actual_tar = sha256(archive)
    if actual_tar != TAR_SHA256:
        print(f'Migration archive checksum mismatch: {actual_tar}', file=sys.stderr)
        return 2

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp.write(archive)
        archive_path = Path(tmp.name)
    try:
        with tarfile.open(archive_path, 'r:gz') as tf:
            safe_extract(tf, root)
    finally:
        archive_path.unlink(missing_ok=True)

    # The transport bundle is intentionally temporary. Remove it before indexing so
    # migration machinery never becomes repository intelligence or long-term source.
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
    run(root, sys.executable, 'scripts/ai-index', 'init')
    run(root, sys.executable, 'scripts/ai-index', 'check')

    print('\nRepository intelligence v2 is installed and regenerated in the working tree.')
    print('No scientific workload or pytest suite was run by this installer.')
    print('Next run: powershell -ExecutionPolicy Bypass -File .\\validation\\validate_ai_repository_intelligence_v2.ps1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

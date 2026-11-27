"""Verify a shared handoff's SHA-256 inventory using the standard library only."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'BUNDLE_MANIFEST.json').read_text())
    for relative, expected in manifest.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or invalid path: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {relative}')
    print(f'PASS: {len(manifest)} distributed files match SHA-256 checksums.')
    print('User-added algorithms, results and downloaded data are not part of the release inventory.')


if __name__ == '__main__':
    verify(Path(__file__).resolve().parents[1])

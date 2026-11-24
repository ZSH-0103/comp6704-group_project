"""Download official datasets, verify archives, and check offline loading."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tarfile
import urllib.request

from torchvision.datasets import FashionMNIST, CIFAR10

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw'


def fetch(url, path, md5=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or (md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5):
        tmp = path.with_name(path.name + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'COMP6704-dataset-setup'})
        with urllib.request.urlopen(request, timeout=90) as response, tmp.open('wb') as output:
            shutil.copyfileobj(response, output)
        if md5 and hashlib.md5(tmp.read_bytes()).hexdigest() != md5:
            raise ValueError(f'Checksum mismatch: {path.name}')
        tmp.replace(path)
    print(f'Available: {path.name}', flush=True)
    return {'url': url, 'file': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'md5_verified': bool(md5)}


def main():
    jobs = [(f'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}',
             RAW / 'FashionMNIST/raw' / name, checksum) for name, checksum in FashionMNIST.resources]
    jobs += [(CIFAR10.url, RAW / CIFAR10.filename, CIFAR10.tgz_md5),
             ('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
              RAW / 'tiny_shakespeare.txt', None)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda args: fetch(*args), jobs))
    for name, _ in FashionMNIST.resources:
        path = RAW / 'FashionMNIST/raw' / name
        with gzip.open(path, 'rb') as source, path.with_suffix('').open('wb') as target:
            shutil.copyfileobj(source, target)
    with tarfile.open(RAW / CIFAR10.filename, 'r:gz') as archive:
        archive.extractall(RAW, filter='data')
    counts = {}
    for cls in (FashionMNIST, CIFAR10):
        counts[cls.__name__] = {}
        for train in (True, False):
            dataset = cls(RAW, train=train, download=False)
            expected = (60000 if cls is FashionMNIST else 50000) if train else 10000
            assert len(dataset) == expected
            sample, label = dataset[0]
            counts[cls.__name__]['train' if train else 'test'] = len(dataset)
            print(f'{cls.__name__} train={train}: {len(dataset)}, image={sample.size}', flush=True)
    text = (RAW / 'tiny_shakespeare.txt').read_text(encoding='utf-8')
    assert len(text) > 1003851 and len(set(text)) == 65
    counts['tiny_shakespeare'] = {'characters': len(text), 'vocab_size': len(set(text))}
    report = {'sources': records, 'offline_load_checks': counts}
    (ROOT / 'data/download-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()

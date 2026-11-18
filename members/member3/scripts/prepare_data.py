"""Download public datasets with checksums, then use Member 4's split code."""
import gzip, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RAW=ROOT/'data/raw'; RAW.mkdir(parents=True,exist_ok=True)
def fetch(url,path,md5=None):
    if path.exists() and (md5 is None or hashlib.md5(path.read_bytes()).hexdigest()==md5): return
    for attempt in range(4):
        part=path.with_suffix(path.suffix+'.part')
        result=subprocess.run(['curl','--http1.1','--fail','--location','--retry','3','--retry-all-errors','--connect-timeout','30','--max-time','600','-o',str(part),url])
        if result.returncode==0 and (md5 is None or hashlib.md5(part.read_bytes()).hexdigest()==md5):
            part.replace(path); return
        print(f'Retrying {path.name}: transfer or checksum failure {attempt+1}',flush=True)
    raise RuntimeError(f'Could not verify {path}')
files=[('train-images-idx3-ubyte.gz','8d4fb7e6c68d591d4c3dfef9ec88bf0d'),('train-labels-idx1-ubyte.gz','25c81989df183df01b3e8a0aad5dffbe'),('t10k-images-idx3-ubyte.gz','bef4ecab320f06d8554ea6380940ec79'),('t10k-labels-idx1-ubyte.gz','bb300cfdad3c16e7a12a480ee83cd310')]
fashion=RAW/'FashionMNIST/raw'; fashion.mkdir(parents=True,exist_ok=True)
for name,md5 in files:
    p=fashion/name
    fetch('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name,p,md5)
    p.with_suffix('').write_bytes(gzip.decompress(p.read_bytes()))
archive=RAW/'cifar-10-python.tar.gz'
expected='c58f30108f718f92721af3b95e74349a'
if not archive.exists() or hashlib.md5(archive.read_bytes()).hexdigest()!=expected:
    # The official server supports byte ranges; bounded concurrency avoids its slow single stream.
    total=170498071; width=(total+15)//16
    parts=ROOT/'build/cifar_download'; parts.mkdir(parents=True,exist_ok=True)
    def chunk(i):
        start=i*width; end=min(total-1,(i+1)*width-1); dest=parts/f'{i:02d}.part'
        if dest.exists() and dest.stat().st_size==end-start+1: return dest
        for attempt in range(3):
            r=subprocess.run(['curl','-sS','--http1.1','--fail','--location','--retry','2','--connect-timeout','30','--max-time','600','--range',f'{start}-{end}','-o',str(dest),'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'])
            if r.returncode==0 and dest.stat().st_size==end-start+1:
                print('CIFAR chunk complete',i,flush=True); return dest
        raise RuntimeError(f'CIFAR chunk {i} failed')
    with ThreadPoolExecutor(max_workers=8) as pool: downloaded=list(pool.map(chunk,range(16)))
    with archive.open('wb') as stream:
        for p in downloaded: stream.write(p.read_bytes())
    if hashlib.md5(archive.read_bytes()).hexdigest()!=expected: raise RuntimeError('CIFAR MD5 mismatch')
fetch('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',RAW/'tiny_shakespeare.txt')
if hashlib.sha256((RAW/'tiny_shakespeare.txt').read_bytes()).hexdigest()!='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed':
    raise RuntimeError('Tiny Shakespeare source differs from the frozen corpus; do not silently change splits')
import torch
from src.common.data import load_task
from src.common.models import make_model
torch.set_num_threads(4)
cfg=json.loads((ROOT/'configs/shared_protocol.json').read_text())
summary={}
for task,tc in cfg['tasks'].items():
    d=load_task(task,cfg,ROOT); m,h=make_model(task,tc,1,d.metadata)
    summary[task]={'sizes':{k:len(getattr(d,k)) for k in ['train','validation','test']},'parameters':sum(p.numel() for p in m.parameters()),'split_hash':d.split_hash,'seed1_model_hash':h,'metadata':d.metadata}
    print(task,summary[task],flush=True)
(ROOT/'data/manifests/dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

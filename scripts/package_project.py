"""Build a portable local archive; does not create or publish a GitHub repository."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'comp6704-optimizer-study'
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'build', 'dist', 'checkpoints', 'node_modules'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.zip', '.pt', '.pth', '.aux', '.log', '.bbl', '.blg', '.fls', '.fdb_latexmk', '.synctex.gz'}

def eligible(p, base):
    rel = p.relative_to(base)
    return (p.is_file() and not p.is_symlink()
        and not any(x in SKIP_DIRS for x in rel.parts)
        and p.name != '.DS_Store' and not p.name.startswith('.env')
        and p.name != 'COMP6704_Optimization_Group_Project_2026_Fall-1.pdf'
        and p.suffix not in SKIP_SUFFIXES
        and not (rel.parts[:2] == ('data','raw') and p.name != '.gitkeep'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist'/'comp6704_project_scaffold.zip')
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    report = args.report_dir or ROOT/'report'
    if not (report/'main.tex').is_file() or not (report/'main.pdf').is_file():
        parser.error('Compile the report first; report source and PDF are required in the archive.')
    files = {}
    for p in sorted(ROOT.rglob('*')):
        if eligible(p, ROOT) and p.resolve() != args.output.resolve() and 'report' not in p.relative_to(ROOT).parts[:1]:
            files[p.relative_to(ROOT).as_posix()] = p
    for p in sorted(report.rglob('*')):
        if eligible(p, report):
            files['report/'+p.relative_to(report).as_posix()] = p
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'scaffold; experimental completion must be recorded by members',
        'files': {name: hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}}
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as z:
        for name,p in files.items():
            z.write(p, PREFIX+'/'+name)
        z.writestr(PREFIX+'/ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    with zipfile.ZipFile(args.output) as z:
        bad=z.testzip()
        if bad:
            raise RuntimeError('Archive integrity failure: '+bad)
    print(f'Created {args.output.resolve()} ({len(files)+1} files, {args.output.stat().st_size} bytes)')

if __name__ == '__main__':
    main()

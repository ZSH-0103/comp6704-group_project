"""Create and verify a portable Member 3 ZIP; no upload or Git mutation."""
import hashlib, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/comp6704_member3_complete.zip'
TOP={'README.md','.gitignore','requirements.in','requirements-lock.txt'}
DIRECTORIES={'configs','src','scripts','tests','docs','results','report','reference','data'}
EXCLUDE={'build','__pycache__','.pytest_cache','.venv','.git','deliverables'}

def include(p):
    r=p.relative_to(ROOT)
    if p.is_symlink() or not p.is_file(): return False
    if any(part in EXCLUDE for part in r.parts): return False
    if r.name=='.DS_Store' or p.suffix in {'.pt','.pyc','.pyo'}: return False
    if r.parts[0]=='data' and r.parts[1]!='manifests': return False
    if r.parts[:2]==('reference','papers'): return False
    return r.name in TOP if len(r.parts)==1 else r.parts[0] in DIRECTORIES

def main():
    validation=json.loads((ROOT/'results/member3/notes/deliverable_validation.json').read_text())
    if validation['status']!='passed': raise RuntimeError('Validate the completed artifact first')
    files=[p for p in sorted(ROOT.rglob('*')) if include(p)]
    entries={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'kind':'completed_member3_module_not_complete_group_submission','files':entries}
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files: z.write(p,'comp6704-member3/'+str(p.relative_to(ROOT)))
        z.writestr('comp6704-member3/ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(OUT) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP integrity check failed')
        for path,entry in entries.items():
            if hashlib.sha256(z.read('comp6704-member3/'+path)).hexdigest()!=entry['sha256']: raise RuntimeError(path)
    checksum=hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix('.zip.sha256').write_text(checksum+'  '+OUT.name+'\n')
    print(json.dumps(dict(zip=str(OUT),files=len(entries),size_bytes=OUT.stat().st_size,sha256=checksum,integrity='verified'),indent=2))

if __name__=='__main__': main()

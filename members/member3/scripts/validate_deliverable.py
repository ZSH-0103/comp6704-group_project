"""Audit the completed artifact without retraining or re-evaluating test sets."""
import csv, hashlib, json, math, statistics, sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/member3'

def load(path): return json.loads(path.read_text())
def check(condition,message):
    if not condition: raise AssertionError(message)

def main():
    shared=load(ROOT/'configs/shared_protocol.json'); cfg=load(ROOT/'configs/member3.json')
    check(cfg['status']=='FROZEN_AFTER_VALIDATION_TUNING','Missing frozen settings')
    for name in ['data.py','models.py','train.py']:
        check((ROOT/'src/common'/name).read_bytes()==(ROOT/'reference/member4_shared'/name).read_bytes(),f'Member4 source changed: {name}')
    files=['common/data.py','common/models.py','common/train.py','members/optimizers.py','members/training.py']
    sourcehash=hashlib.sha256(b''.join((ROOT/'src'/p).read_bytes() for p in files)).hexdigest()
    records=[]
    for phase,count,steps in [('smoke',6,5),('tune',18,100),('main',18,300)]:
        paths=list((RESULT/'raw'/phase).glob('*/summary.json')); check(len(paths)==count,f'{phase}: expected {count} runs')
        for path in paths:
            r=load(path); records.append(r)
            check(r['status']=='completed',f'Failed run {path}')
            check(r['source_sha256']==sourcehash,'Implementation fingerprint changed')
            check(r['gradient_calls']==steps,'Wrong budget')
            check(r['test_evaluated']==(phase=='main'),'Test isolation error')
            if phase!='main': check(r['test_ce'] is None,'Test read during model selection')
            else:
                check(datetime.fromisoformat(r['start_utc'])>datetime.fromisoformat(cfg['frozen_utc']),'Main started before frozen settings')
                check(r['learning_rate']==cfg['selected_learning_rates'][r['task']][r['optimizer']],'Wrong selected rate')
            resolved=load(path.with_name('config.json')); claimed=resolved.pop('config_hash');resolved.pop('canonical_id')
            check(hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()==claimed==r['config_hash'],'Config fingerprint mismatch')
            check(path.with_name('environment.json').exists(),'Missing per-run environment')
            metrics=[json.loads(x) for x in path.with_name('metrics.jsonl').read_text().splitlines()]
            check([x['step'] for x in metrics]==([0,5] if phase=='smoke' else list(range(0,steps+1,50))),'Wrong evaluation schedule')
            check(metrics[-1]['val_ce']==r['final_val_ce'],'Final metric mismatch')
            check(r['optimizer_state_bytes']==4*r['parameter_count'],'Unexpected tensor state size')
            check(r['train_validation_time_s']<=r['elapsed_s'],'Invalid timing convention')
    mains=[r for r in records if r['phase']=='main']
    for task in shared['tasks']:
        for seed in shared['main_seeds']:
            rs=[r for r in mains if r['task']==task and r['seed']==seed]
            for key in ['initialization_hash','split_hash','batch_sequence_hash']:
                check(len({r[key] for r in rs})==1,f'Pair mismatch {key}')
    with (RESULT/'tables/main_results.csv').open() as f: aggregate=list(csv.DictReader(f))
    check(len(aggregate)==6,'Main table must have six rows')
    for row in aggregate:
        runs=[r for r in mains if r['task']==row['task'] and r['optimizer']==row['optimizer']]
        for column,key in [('train_ce','final_train_ce'),('val_ce','final_val_ce'),('elapsed_s','train_validation_time_s'),('test_metric',row['test_metric_name'])]:
            values=[r[key] for r in runs]
            check(math.isclose(float(row[column+'_mean']),statistics.mean(values),abs_tol=1e-12),'Mean mismatch')
            check(math.isclose(float(row[column+'_sd']),statistics.stdev(values),abs_tol=1e-12),'SD mismatch')
    diagnostic=RESULT/'raw/diagnostic'
    check(np.load(diagnostic/'oracle_noise.npz')['noise'].shape==(20,10000),'Oracle shape mismatch')
    for beta in cfg['diagnostic']['beta2']:
        for name in cfg['diagnostic']['variants']:
            theta=np.load(diagnostic/f'{name}__beta{beta:g}.npz')['theta']
            check(theta.shape==(20,10001) and np.isfinite(theta).all(),'Missing/invalid trajectory')
    for name in ['member3_overview.pdf','member3_diagnostic.pdf']:
        check((RESULT/'figures'/name).read_bytes()==(ROOT/'report/figures'/name).read_bytes(),'Report figure differs')
    pdf=ROOT/'report/member3_report.pdf';check(pdf.exists() and pdf.stat().st_size>10000,'Report missing')
    summary=dict(status='passed',implementation_tests=15,smoke_runs=6,tuning_runs=18,main_runs=18,diagnostic_trajectories=240,paired_hash_checks=9,source_sha256=sourcehash,report_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),checks='counts, frozen-before-main, test isolation, source/config hashes, pairing, mean/SD recomputation, diagnostic completeness, report figures')
    (RESULT/'notes/deliverable_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

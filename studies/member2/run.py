"""Reproduce the final library-based Momentum/Schedule-Free study."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
import uuid

import torch
from schedulefree import SGDScheduleFree
from .runtime.data import load_task
from .runtime.models import make_model
from .runtime.train import train_one

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configuration():
    cfg = json.loads((ROOT/'configs/member2.json').read_text())
    cfg['shared'] = json.loads((ROOT/cfg['shared_protocol']).read_text())
    return cfg


def cosine(step, horizon):
    return (1 + math.cos(math.pi * min(step, horizon) / horizon)) / 2


def execute(phase, cfg, tasks=None, seeds=None):
    """Write new timestamped results; never overwrite published observations."""
    shared = cfg['shared']
    torch.set_num_threads(shared['torch_threads'])
    dest = HOME/'runs'/(phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest/'resolved_config.json', cfg)
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'threads': torch.get_num_threads(), 'device': 'cpu',
                   'versions': {p: version(p) for p in ['torch','torchvision','schedulefree','numpy']}}
    entries = []
    tasks = tasks or cfg['tasks']
    seeds = seeds or ([11] if phase == 'smoke' else shared['main_seeds'])
    for task in tasks:
        if phase == 'diagnostic' and task != cfg['diagnostic']['task']:
            continue
        data = load_task(task, shared, HOME)
        tc = shared['tasks'][task]
        horizons = cfg['diagnostic']['horizons'] if phase == 'diagnostic' else [5 if phase == 'smoke' else tc['max_updates']]
        methods = cfg['diagnostic']['methods'] if phase == 'diagnostic' else cfg['optimizers']
        for horizon in horizons:
            for method in methods:
                base = 'schedule_free_sgd' if method == 'schedule_free_sgd' else 'momentum'
                for seed in seeds:
                    model, initial_hash = make_model(task, tc, seed, data.metadata)
                    lr = cfg['learning_rates'][task][base]
                    if base == 'momentum':
                        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'], weight_decay=0)
                        h = cfg['diagnostic']['planned_horizon'] if method == 'momentum_fixed_T' else horizon
                        schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: cosine(step, h))
                    else:
                        opt = SGDScheduleFree(model.parameters(), lr=lr, momentum=cfg['momentum'], warmup_steps=cfg['sf_warmup_steps'], weight_decay=0)
                        schedule = None
                    run_id = f'{phase}_{task}_{method}_s{seed}_h{horizon}'
                    folder = dest/'raw'/run_id
                    folder.mkdir(parents=True)
                    record = {'phase': phase, 'task': task, 'optimizer': method, 'seed': seed,
                              'updates': horizon, 'learning_rate': lr, 'initialization_hash': initial_hash,
                              'split_hash': data.split_hash, 'schedule_horizon': h if schedule else None,
                              'batch_size': tc['batch_size'], 'eval_examples': cfg['eval_examples'],
                              'batch_generator_seed': seed+100000, 'parameter_mode': 'train',
                              'test_evaluations_expected': int(phase=='main')}
                    write_json(folder/'config.json', record)
                    write_json(folder/'environment.json', environment)
                    with (folder/'metrics.jsonl').open('w') as handle:
                        def log(row):
                            handle.write(json.dumps(row, allow_nan=False)+'\n')
                            handle.flush()
                        _, summary = train_one(model, opt, data, task, seed, horizon, tc['batch_size'],
                                               min(tc['eval_interval'], horizon), None,
                                               eval_examples=cfg['eval_examples'], evaluate_test=phase=='main',
                                               scheduler=schedule, on_record=log)
                    torch.save({'model':model.state_dict(), 'optimizer':opt.state_dict(),
                                'scheduler':schedule.state_dict() if schedule else None,
                                'config':record, 'step':horizon, 'parameter_mode':'train'}, folder/'checkpoint.pt')
                    summary.update(record, status='completed')
                    write_json(folder/'summary.json',summary)
                    entries.append(dict(directory='raw/'+run_id, **record))
                    write_json(dest/'manifest.json', {'runs':entries})
                    print(f'{run_id}: val CE={summary["final_val_ce"]:.6f}', flush=True)
    if not entries:
        raise ValueError('No runs selected (diagnostics use Fashion-MNIST only)')
    print(dest)
    return dest


def records(source, verify=False):
    entries = json.loads((source/'manifest.json').read_text())['runs']
    result = []
    for e in entries:
        folder = source/e['directory']
        if verify:
            for name, expected in e['sha256'].items():
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected, folder/name
        s = json.loads((folder/'summary.json').read_text())
        rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        result.append(({**s, **e}, rows))
    return result


def check():
    runs = records(HOME/'published', verify=True)
    assert sum(s['phase']=='main' for s,_ in runs)==18
    assert sum(s['phase']=='diagnostic' for s,_ in runs)==27
    cfg = configuration()
    pairs = {}
    for s, rows in runs:
        assert s['status']=='completed'
        assert rows[-1]['step']==s['updates']
        assert all(math.isfinite(r['val_ce']) for r in rows)
        folder=HOME/'published'/s['directory']
        c=json.loads((folder/'config.json').read_text())
        key=(s['task'],s['seed'])
        assert pairs.setdefault(key,(c['initialization_hash'],c['split_hash']))==(c['initialization_hash'],c['split_hash'])
        base='schedule_free_sgd' if s['optimizer']=='schedule_free_sgd' else 'momentum'
        assert s['learning_rate']==cfg['learning_rates'][s['task']][base]
        assert s['target_val_ce'] is None
        if s['phase']=='main': assert s['test_ce'] is not None
        else: assert s['test_ce'] is None
        if s['optimizer']=='momentum_fixed_T' and s['updates']==600:
            assert all(r['learning_rate']==0 for r in rows if r['step']>=300)
            assert abs(rows[6]['val_ce']-rows[-1]['val_ce'])<1e-7
    print('Verified 45 published runs: hashes, final rates, paired splits/initialization, test isolation and fixed-horizon behavior.')


def export(source, output):
    """Regenerate tables/plots solely from the chosen measured run archive."""
    import numpy as np
    os.environ.setdefault('MPLCONFIGDIR',str(HOME/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=records(source)
    output.mkdir(parents=True,exist_ok=True)
    main=[]; diagnostic=[]
    for phase in ['main','diagnostic']:
        groups={}
        for s,rows in runs:
            if s['phase']==phase:
                groups.setdefault((s['task'],s['optimizer'],s['updates']),[]).append((s,rows))
        for (task,method,updates),group in groups.items():
            row=dict(task=task,optimizer=method,updates=updates,n=len(group),learning_rate=group[0][0]['learning_rate'])
            for metric in ['final_train_ce','final_val_ce','test_ce','test_accuracy_pct','elapsed_s','optimizer_state_bytes']:
                vals=[s[metric] for s,_ in group if s.get(metric) is not None]
                row[metric+'_mean']=statistics.mean(vals) if vals else None
                row[metric+'_sd']=statistics.stdev(vals) if len(vals)>1 else None
            (main if phase=='main' else diagnostic).append(row)
    for name,rows in [('main_results',main),('diagnostic_results',diagnostic)]:
        if rows:
            with (output/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels={'momentum':'Momentum','schedule_free_sgd':'Schedule-Free SGD','momentum_fixed_T':'Momentum: planned T=300','momentum_horizon_aware':'Momentum: actual horizon'}
    if main:
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for j,task in enumerate(configuration()['tasks']):
            for method in ['momentum','schedule_free_sgd']:
                traces=[rows for s,rows in runs if s['phase']=='main' and s['task']==task and s['optimizer']==method]
                if not traces:continue
                ys=np.array([[r['val_ce'] for r in t] for t in traces]);mean=ys.mean(0);sd=ys.std(0,ddof=1) if len(traces)>1 else np.zeros_like(mean)
                for ax,x in zip(axes[:,j],[[r['step'] for r in traces[0]],np.mean([[r['elapsed_s'] for r in t] for t in traces],axis=0)]):
                    ax.plot(x,mean,label=labels[method]);ax.fill_between(x,mean-sd,mean+sd,alpha=.15);ax.grid(alpha=.2);ax.set_ylabel('Validation CE');ax.legend(fontsize=8)
            axes[0,j].set_title(task);axes[0,j].set_xlabel('Updates');axes[1,j].set_xlabel('Mean elapsed time (s)')
        fig.savefig(output/'member2_main.pdf');plt.close(fig)
    if diagnostic:
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        for method in configuration()['diagnostic']['methods']:
            rows=sorted([r for r in diagnostic if r['optimizer']==method],key=lambda r:r['updates'])
            ax.errorbar([r['updates'] for r in rows],[r['final_val_ce_mean'] for r in rows],yerr=[r['final_val_ce_sd'] or 0 for r in rows],marker='o',capsize=4,label=labels[method])
        ax.set(xlabel='Actual training updates',ylabel='Validation CE',title='Fashion-MNIST stopping-horizon comparison');ax.legend();ax.grid(alpha=.2)
        fig.savefig(output/'member2_diagnostic.pdf');plt.close(fig)
    print(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['check','smoke','main','diagnostic','export'])
    p.add_argument('--task',choices=list(configuration()['tasks']))
    p.add_argument('--seed',type=int)
    p.add_argument('--source',type=Path,default=HOME/'published')
    p.add_argument('--output',type=Path,default=HOME/'generated')
    a=p.parse_args()
    if a.phase=='check':check()
    elif a.phase=='export':export(a.source,a.output)
    else:execute(a.phase,configuration(),[a.task] if a.task else None,[a.seed] if a.seed is not None else None)


if __name__=='__main__':main()

"""Validate complete logs and export measured tables and publication figures."""
import csv, hashlib, json, math, shutil, statistics, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/member3'
SHARED=json.loads((ROOT/'configs/shared_protocol.json').read_text())
M3=json.loads((ROOT/'configs/member3.json').read_text())
SUCCESS=('success','completed')
LABELS={'fashion_mnist':'Fashion-MNIST','cifar10':'CIFAR-10','tiny_shakespeare':'Tiny Shakespeare','rmsprop':'RMSProp','adopt_no_momentum':'ADOPT-0','current_moment_matched_clip':'Current + matched clip','adopt_no_clip':'ADOPT-0, no clip'}

def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)

def save(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def mean(x): return statistics.mean(x) if x else None
def sd(x): return statistics.stdev(x) if len(x)>1 else None

def export():
    if M3['status']!='FROZEN_AFTER_VALIDATION_TUNING': raise RuntimeError('Hyperparameters are not frozen')
    allruns=[]
    env=json.loads((RESULT/'environment.json').read_text())
    hw=RESULT/'hardware.json'
    if hw.exists(): env.update(json.loads(hw.read_text()))
    save(RESULT/'environment.json',env)
    for path in sorted((RESULT/'raw').glob('*/*/summary.json')):
        r=json.loads(path.read_text()); out=path.parent
        resolved=dict(member=3,phase=r['phase'],task=r['task'],seed=r['seed'],shared_protocol=SHARED,
                      optimizer=dict(name=r['optimizer'],learning_rate=r['learning_rate'],beta1=0,beta2=r['beta2'],epsilon=r['epsilon'],weight_decay=0,clip=r['optimizer']=='adopt_no_momentum'),
                      gradient_call_budget=r['max_updates'],split_hash=r['split_hash'],initialization_hash=r['initialization_hash'],implementation_sha256=r['source_sha256'],test_evaluation=r['phase']=='main')
        h=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
        resolved.update(config_hash=h,canonical_id=f"m3__{r['phase']}__{r['run_id']}__{h[:12]}")
        save(out/'config.json',resolved);save(out/'environment.json',env)
        r.update(config_hash=h,error_message=r.get('error'),status='completed' if r['status'] in SUCCESS else 'error')
        save(path,r)
        metrics=out/'metrics.csv'
        if metrics.exists():
            rows=read_csv(metrics)
            with (out/'metrics.jsonl').open('w') as f:
                for row in rows:
                    row={k:(None if v=='' else int(v) if k=='step' else float(v)) for k,v in row.items()}
                    f.write(json.dumps(row,allow_nan=False)+'\n')
        r['_path']=out; allruns.append(r)
    main=[r for r in allruns if r['phase']=='main']; tune=[r for r in allruns if r['phase']=='tune']
    if len(main)!=18 or len(tune)!=18: raise RuntimeError(f'Expected 18 main + 18 tuning runs, got {len(main)}, {len(tune)}')
    rows=[];cost=[];paired=[]
    for task in SHARED['tasks']:
        for seed in SHARED['main_seeds']:
            pair=[r for r in main if r['task']==task and r['seed']==seed]
            if len(pair)!=2: raise RuntimeError('Incomplete pairing')
            for key in ['initialization_hash','split_hash','batch_sequence_hash']:
                if len({r.get(key) for r in pair})!=1: raise RuntimeError(f'Pair mismatch: {task}/{seed}/{key}')
        for name in M3['optimizers']:
            rs=[r for r in main if r['task']==task and r['optimizer']==name]
            if sorted(r['seed'] for r in rs)!=SHARED['main_seeds']: raise RuntimeError('Missing or duplicate seeds')
            if any(r['learning_rate']!=M3['selected_learning_rates'][task][name] for r in rs): raise RuntimeError('Wrong frozen LR')
            good=[r for r in rs if r['status']=='completed']; reached=[r for r in good if r['target_reached']]
            row=dict(task=task,optimizer=name,n_expected=3,n_completed=len(good))
            for col,key in [('train_ce','final_train_ce'),('val_ce','final_val_ce')]:
                values=[r[key] for r in good];row[col+'_mean']=mean(values);row[col+'_sd']=sd(values)
            key='test_ce' if task=='tiny_shakespeare' else 'test_accuracy_pct'
            values=[r[key] for r in good]
            row.update(test_metric_name=key,test_metric_mean=mean(values),test_metric_sd=sd(values),elapsed_s_mean=mean([r['train_validation_time_s'] for r in good]),elapsed_s_sd=sd([r['train_validation_time_s'] for r in good]),total_including_test_s_mean=mean([r['elapsed_s'] for r in good]),learning_rate=M3['selected_learning_rates'][task][name])
            rows.append(row)
            ts=[r for r in tune if r['task']==task and r['optimizer']==name]
            cost.append(dict(task=task,optimizer=name,n_expected=3,n_completed=len(good),n_target_reached=len(reached),target_val_ce=SHARED['tasks'][task]['target_val_ce'],target_steps_mean_reached=mean([r['target_step'] for r in reached]),target_time_s_mean_reached=mean([r['target_time_s'] for r in reached]),optimizer_state_bytes=good[0]['optimizer_state_bytes'] if good else None,tuning_trials=len(ts),tuning_elapsed_s=sum(r.get('elapsed_s',0) for r in ts),gradient_calls=300,parameter_updates=good[0]['parameter_updates'] if good else None))
        for metric in ['final_val_ce','test_ce' if task=='tiny_shakespeare' else 'test_accuracy_pct','train_validation_time_s']:
            diff=[]
            for seed in SHARED['main_seeds']:
                pair={r['optimizer']:r for r in main if r['task']==task and r['seed']==seed and r['status']=='completed'}
                if len(pair)==2: diff.append(pair['adopt_no_momentum'][metric]-pair['rmsprop'][metric])
            paired.append(dict(task=task,metric=metric,direction='ADOPT0 minus RMSProp',n=len(diff),mean=mean(diff),sd=sd(diff)))
    write_csv(RESULT/'tables/main_results.csv',rows);write_csv(RESULT/'tables/cost_stability.csv',cost);write_csv(RESULT/'tables/paired_differences.csv',paired)
    tuning=[]
    for r in tune:
        tuning.append(dict(trial_id=r['run_id'],task=r['task'],optimizer=r['optimizer'],seed=r['seed'],hyperparameters_json=json.dumps(dict(lr=r['learning_rate'],beta2=r['beta2'],eps=r['epsilon'])),validation_ce=r.get('final_val_ce'),status=r['status'],elapsed_s=r.get('elapsed_s'),selected=r['learning_rate']==M3['selected_learning_rates'][r['task']][r['optimizer']]))
    write_csv(RESULT/'notes/tuning_log.csv',tuning)
    diag=read_csv(RESULT/'raw/diagnostic/diagnostic_runs.csv'); dr=[]
    if len(diag)!=240: raise RuntimeError('Incomplete diagnostic')
    for beta in M3['diagnostic']['beta2']:
        for name in M3['diagnostic']['variants']:
            rs=[r for r in diag if float(r['beta2'])==beta and r['variant']==name]
            if len(rs)!=20 or any(r['status']!='success' for r in rs): raise RuntimeError('Diagnostic missing/failed runs')
            config_hash=hashlib.sha256(json.dumps(dict(M3['diagnostic'],selected_beta2=beta,selected_variant=name),sort_keys=True).encode()).hexdigest()
            for key,unit in [('tail_mean_objective','objective gap'),('final_objective','objective gap'),('tail_mean_theta','parameter')]:
                values=[float(r[key]) for r in rs]
                dr.append(dict(diagnostic_id='logcosh_asymmetric_oracle',condition=f'beta2={beta}',optimizer=name,metric_name=key,metric_unit=unit,n=len(rs),mean=mean(values),sd=sd(values),failures=0,config_hash=config_hash))
    write_csv(RESULT/'tables/diagnostic_results.csv',dr)
    normrows=[]
    for beta in M3['diagnostic']['beta2']:
        for name in M3['diagnostic']['variants']:
            h=np.load(RESULT/f'raw/diagnostic/{name}__beta{beta:g}.npz')['theta']
            for i,seed in enumerate(M3['diagnostic']['seeds']):
                normrows.append(dict(variant=name,beta2=beta,seed=seed,tail_mean_squared_true_gradient=float(np.square(np.tanh(h[i,-1000:])).mean()),tail_mean_absolute_update=float(np.abs(np.diff(h[i])[-1000:]).mean())))
    write_csv(RESULT/'tables/diagnostic_gradient_update_norms.csv',normrows)
    print('Exported measured tables; verified 9 paired initialization/split/batch hashes.')
    return main

def figures(main):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'rmsprop':'#245E9B','adopt_no_momentum':'#C65B22','current_moment_matched_clip':'#638D5A','adopt_no_clip':'#9467BD'}
    fig,axes=plt.subplots(3,2,figsize=(8,7),constrained_layout=True)
    for i,task in enumerate(SHARED['tasks']):
        for name in M3['optimizers']:
            rs=[r for r in main if r['task']==task and r['optimizer']==name and r['status']=='completed']
            series=[read_csv(r['_path']/'metrics.csv') for r in rs]
            steps=np.array([float(x['step']) for x in series[0]])
            val=np.array([[float(x['val_ce']) for x in s] for s in series])
            train=np.array([[float(x['train_ce']) for x in s] for s in series])
            c=colors[name]; a=axes[i,0]
            a.plot(steps,val.mean(0),label=LABELS[name],c=c); a.fill_between(steps,val.mean(0)-val.std(0,ddof=1),val.mean(0)+val.std(0,ddof=1),alpha=.16,color=c)
            a.plot(steps,train.mean(0),c=c,ls=':',lw=1)
            times=[np.array([float(x['elapsed_s']) for x in s]) for s in series]
            grid=np.linspace(max(t[0] for t in times),min(t[-1] for t in times),100)
            aligned=np.stack([np.interp(grid,t,v) for t,v in zip(times,val)])
            b=axes[i,1]; b.plot(grid,aligned.mean(0),label=LABELS[name],c=c)
            b.fill_between(grid,aligned.mean(0)-aligned.std(0,ddof=1),aligned.mean(0)+aligned.std(0,ddof=1),alpha=.16,color=c)
        axes[i,0].set_title(LABELS[task]);axes[i,1].set_title(LABELS[task]+' — validation')
        axes[i,0].set_xlabel('Gradient calls');axes[i,1].set_xlabel('Elapsed time (s)')
        for a in axes[i]: a.set_ylabel('Cross-entropy');a.grid(alpha=.2)
    axes[0,0].legend();axes[0,1].legend();fig.suptitle('Member 3: means ± sample SD, 3 seeds; dotted lines = train CE',fontsize=11)
    out=RESULT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'member3_overview.pdf');fig.savefig(out/'member3_overview.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
    curves=read_csv(RESULT/'raw/diagnostic/diagnostic_curves.csv')
    diag=read_csv(RESULT/'tables/diagnostic_results.csv')
    for j,name in enumerate(M3['diagnostic']['variants']):
        rs=[r for r in curves if r['variant']==name and float(r['beta2'])==.9]
        x=np.array([float(r['step']) for r in rs]); y=np.array([float(r['theta_mean']) for r in rs])
        axes[0].plot(x,y,label=LABELS[name],color=colors[name],ls='-' if j<2 else '--',lw=1.4 if j<2 else 1.)
        ds=[next(r for r in diag if r['condition']==f'beta2={b}' and r['optimizer']==name and r['metric_name']=='tail_mean_objective') for b in M3['diagnostic']['beta2']]
        positions=np.arange(3)+(j-1.5)*.19
        axes[1].bar(positions,[float(r['mean']) for r in ds],width=.18,color=colors[name],yerr=[float(r['sd']) for r in ds],capsize=2,label=LABELS[name],alpha=.85)
    axes[0].axhline(0,c='black',lw=.7,ls=':');axes[0].set(xlabel='Gradient calls',ylabel='Mean parameter θ',title='Asymmetric noise, β₂ = 0.9');axes[0].legend(fontsize=7)
    axes[1].set_xticks(range(3),['0.9','0.99','0.999']);axes[1].set(xlabel='β₂',ylabel='Tail-average objective gap',title='Last 1,000 calls; 20 seeds, mean ± SD')
    fig.savefig(out/'member3_diagnostic.pdf');fig.savefig(out/'member3_diagnostic.png',dpi=180);plt.close(fig)
    destination=ROOT/'report/figures';destination.mkdir(parents=True,exist_ok=True)
    for name in ['member3_overview.pdf','member3_diagnostic.pdf']: shutil.copy2(out/name,destination/name)

if __name__=='__main__': figures(export())

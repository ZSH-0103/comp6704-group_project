"""Generate Member 1 report tables/figures from saved formal/diagnostic evidence."""
import csv
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR','/tmp/comp6704-member1-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'
RESULT=ROOT/'results/member1'

def main():
    generated=OUT/'generated';generated.mkdir(exist_ok=True)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    rows=list(csv.DictReader((RESULT/'tables/main_results.csv').open()))
    manifest=json.loads((RESULT/json.loads((RESULT/'notes/main_latest.json').read_text())['manifest']).read_text())
    assert len(rows)==6 and len(manifest['runs'])==18
    # Independently verify table numbers against the raw per-run summaries.
    for row in rows:
        group=[json.loads((RESULT/'raw'/r['run_id']/'summary.json').read_text()) for r in manifest['runs'] if r['task']==row['task'] and r['optimizer']==row['optimizer']]
        assert len(group)==3 and all(r['status']=='completed' for r in group)
        for field,prefix in [('final_train_ce','train_ce'),('final_val_ce','val_ce'),('test_ce' if row['task']=='tiny_shakespeare' else 'test_accuracy_pct','test_metric'),('elapsed_s','elapsed_s')]:
            values=[r[field] for r in group]
            assert np.isclose(float(row[prefix+'_mean']),statistics.mean(values),rtol=1e-13)
            assert np.isclose(float(row[prefix+'_sd']),statistics.stdev(values),rtol=1e-13)
    def pm(row,prefix,digits):
        return f"${float(row[prefix+'_mean']):.{digits}f}\\pm{float(row[prefix+'_sd']):.{digits}f}$"
    text=['% Generated from measured main_results.csv; do not edit numbers manually.',r'\begin{table}[H]\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',
          r'\caption{M1-MAIN (Member 1): mean $\pm$ sample SD over seeds 1--3; all runs completed. Train/validation are CE; test quality is accuracy (\%) for images and CE (nats/character) for Text. Time includes training and scheduled evaluation, excluding final test and tuning.}',r'\label{tab:m1-main}',r'\begin{tabular}{llrrrrr}\toprule',r'Task & Method & Train CE & Val CE & Test quality & Time (s) & $n$\\\midrule']
    names={'fashion_mnist':'Fashion','cifar10':'CIFAR','tiny_shakespeare':'Text'}
    for row in rows:
        cells=[names[row['task']],row['optimizer'].upper() if row['optimizer']=='sgd' else 'DoG',pm(row,'train_ce',3),pm(row,'val_ce',3),pm(row,'test_metric',4 if row['task']=='tiny_shakespeare' else 2),pm(row,'elapsed_s',2),'3/3']
        text.append(' & '.join(cells)+r'\\')
    text += [r'\bottomrule\end{tabular}\end{table}']
    (generated/'member1_main_table.tex').write_text('\n'.join(text)+'\n')
    diag=list(csv.DictReader((RESULT/'tables/diagnostic_results.csv').open()))
    def get(condition,opt,metric):
        return next(r for r in diag if r['condition']==condition and r['optimizer']==opt and r['metric_name']==metric)
    def sci(row):
        mean,sd=float(row['mean']),float(row['sd'])
        if mean==0: return '$0\\pm0$'
        exp=int(np.floor(np.log10(mean)));factor=10.**exp
        if sd > 0 and sd/factor < 0.005:
            se=int(np.floor(np.log10(sd)))
            return f'${mean/factor:.2f}\\,10^{{{exp}}}\\pm {sd/(10.**se):.2f}\\,10^{{{se}}}$'
        return f'$({mean/factor:.2f}\\pm{sd/factor:.2f})\\,10^{{{exp}}}$'
    text=['% Generated from measured diagnostic_results.csv.',r'\begin{table}[H]\centering\footnotesize',r'\caption{M1-DIAG (Member 1): final relative gap $f(x_{500})/f(x_0)$, mean $\pm$ sample SD across three starting directions. Only DoG at $\kappa=100$ failed to reach $10^{-6}$; every run completed.}',r'\label{tab:m1-diag}',r'\begin{tabular}{rrlll}\toprule',r'$\kappa$ & $a$ & Tuned SGD & DoG & DoG reached\\\midrule']
    levels=[(1,1),(10,1),(100,1),(10,.1),(10,10)]
    for k,a in levels:
        cid=f'k{k:g}_a{a:g}'
        s,d=get(cid,'sgd','final_relative_gap'),get(cid,'dog','final_relative_gap')
        assert int(s['n'])==int(d['n'])==3
        text.append(f'{k} & {a:g} & {sci(s)} & {sci(d)} & '+('0/3' if k==100 else '3/3')+r'\\')
    text += [r'\bottomrule\end{tabular}\end{table}']
    (generated/'member1_diagnostic_table.tex').write_text('\n'.join(text)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.spines.top':False,'axes.spines.right':False})
    colors={'sgd':'#2166a5','dog':'#d66a21'}
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.05))
    for ax,task in zip(axes,names):
        for opt in ['sgd','dog']:
            runs=[r for r in manifest['runs'] if r['task']==task and r['optimizer']==opt]
            data=[[json.loads(l) for l in (RESULT/'raw'/r['run_id']/'metrics.jsonl').read_text().splitlines()] for r in runs]
            ys=np.array([[v['val_ce'] for v in tr] for tr in data]);xs=[v['step'] for v in data[0]]
            mu,sd=ys.mean(0),ys.std(0,ddof=1)
            ax.plot(xs,mu,color=colors[opt],lw=1.2,label='SGD' if opt=='sgd' else 'DoG')
            ax.fill_between(xs,mu-sd,mu+sd,color=colors[opt],alpha=.17)
        ax.axhline(manifest['shared']['tasks'][task]['target_val_ce'],ls=':',color='gray',lw=.7)
        ax.set(title=names[task],xlabel='Updates');ax.grid(alpha=.15);ax.legend(fontsize=6.5,frameon=False)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel('Validation CE')
    fig.subplots_adjust(left=.075,right=.995,bottom=.23,top=.87,wspace=.32)
    for ext in ['pdf','png']: fig.savefig(figures/f'member1_convergence.{ext}',dpi=200)
    plt.close(fig)
    qp=RESULT/json.loads((RESULT/'notes/quadratic_latest.json').read_text())['manifest']
    qm=json.loads(qp.read_text())
    def qdata(r):return [json.loads(l) for l in (RESULT/'raw'/r['run_id']/'metrics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(6.6,1.95))
    for opt in ['sgd','dog']:
        group=[r for r in qm['runs'] if r['kappa']==100 and r['scale']==1 and r['optimizer']==opt]
        ys=np.array([[x['relative_gap'] for x in qdata(r)] for r in group]);xs=np.arange(501)
        mu,sd=ys.mean(0),ys.std(0,ddof=1)
        axes[0].semilogy(xs,np.maximum(mu,1e-16),color=colors[opt],lw=1.2,label='SGD' if opt=='sgd' else 'DoG')
        axes[0].fill_between(xs,np.maximum(mu-sd,1e-16),np.maximum(mu+sd,1e-16),color=colors[opt],alpha=.17)
    axes[0].axhline(1e-6,ls=':',color='gray',lw=.7)
    axes[0].set(title=r'Conditioning: $\kappa=100$, $a=1$',ylabel='Relative objective gap',xlabel='Updates')
    for a,color,style in [(.1,'#17806d','-'),(1,'#5a65ad','--'),(10,'#a45b3d','-.')]:
        group=[r for r in qm['runs'] if r['kappa']==10 and r['scale']==a and r['optimizer']=='dog']
        ys=np.array([[x['learning_rate'] for x in qdata(r)[1:]] for r in group])
        axes[1].semilogy(np.arange(1,501),ys.mean(0),style,color=color,lw=1.1,label=f'a={a:g}')
    axes[1].set(title=r'DoG scaling: $\kappa=10$',ylabel=r'Raw step size $\eta_t$',xlabel='Updates')
    for ax in axes: ax.legend(fontsize=6.5,frameon=False);ax.grid(alpha=.15);ax.tick_params(labelsize=6.5)
    fig.subplots_adjust(left=.09,right=.99,bottom=.24,top=.86,wspace=.43)
    for ext in ['pdf','png']:fig.savefig(figures/f'member1_mechanism.{ext}',dpi=200)
    plt.close(fig)
    audit=dict(main_rows_verified=6,main_raw_runs_verified=18,diagnostic_conditions=5,diagnostic_metric='final_relative_gap',main_batch=manifest['batch'],quadratic_batch=qm['batch'])
    (generated/'member1_numeric_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()

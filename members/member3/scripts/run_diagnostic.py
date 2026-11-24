"""Unconstrained normalization diagnostic: bounded asymmetric unbiased oracle."""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def trajectory(noise, variant, beta, lr, initial=2., eps=1e-8):
    theta=np.full(noise.shape[0],initial,dtype=np.float64)
    v=np.zeros_like(theta); history=np.empty((noise.shape[0],noise.shape[1]+1)); history[:,0]=theta
    for t in range(noise.shape[1]):
        g=np.tanh(theta)+noise[:,t]
        if variant=='rmsprop':
            v=beta*v+(1-beta)*g*g
            theta-=lr*g/(np.sqrt(v)+eps)
        elif t==0:
            v=g*g
        else:
            if variant=='current_moment_matched_clip': v=beta*v+(1-beta)*g*g
            z=g/np.maximum(np.sqrt(v),eps)
            if variant!='adopt_no_clip': z=np.clip(z,-t**.25,t**.25)
            theta-=lr*z
            if variant!='current_moment_matched_clip': v=beta*v+(1-beta)*g*g
        history[:,t+1]=theta
    return history

def main():
    cfg=json.loads((ROOT/'configs/member3.json').read_text())['diagnostic']
    seeds=cfg['seeds']; steps=cfg['gradient_calls']; tail=cfg['tail_window']
    noise=np.stack([np.where(np.random.default_rng(s).random(steps)<.9,-1.,9.) for s in seeds])
    rows=[]; curves=[]
    dest=ROOT/'results/member3/raw/diagnostic'; dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'oracle_noise.npz',noise=noise,seeds=seeds)
    for beta in cfg['beta2']:
        for variant in cfg['variants']:
            history=trajectory(noise,variant,beta,cfg['learning_rate'],cfg['initial_theta'])
            objective=np.logaddexp(history,-history)-np.log(2.)
            np.savez_compressed(dest/f'{variant}__beta{beta:g}.npz',theta=history,seeds=seeds)
            for i,seed in enumerate(seeds):
                rows.append(dict(variant=variant,beta2=beta,seed=seed,status='success' if np.isfinite(history[i]).all() else 'failed',final_objective=objective[i,-1],tail_mean_objective=objective[i,-tail:].mean(),final_theta=history[i,-1],tail_mean_theta=history[i,-tail:].mean(),gradient_calls=steps,learning_rate=cfg['learning_rate']))
            for step in range(0,steps+1,50):
                curves.append(dict(variant=variant,beta2=beta,step=step,objective_mean=objective[:,step].mean(),objective_sd=objective[:,step].std(ddof=1),theta_mean=history[:,step].mean(),theta_sd=history[:,step].std(ddof=1)))
    for filename,records in [('diagnostic_runs.csv',rows),('diagnostic_curves.csv',curves)]:
        with (dest/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    print('Saved',len(rows),'diagnostic trajectories; no learned-model test data used.')

if __name__=='__main__': main()

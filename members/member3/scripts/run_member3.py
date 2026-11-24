"""Run smoke, validation-only tuning and frozen main experiments sequentially."""
import argparse, csv, hashlib, json, platform, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch, torchvision
from src.common.data import load_task
from src.common.models import make_model
from src.members.optimizers import make_optimizer
from src.members.training import train_one

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['smoke','tune','main'])
    parser.add_argument('--task',choices=['fashion_mnist','cifar10','tiny_shakespeare'])
    args=parser.parse_args()
    shared=json.loads((ROOT/'configs/shared_protocol.json').read_text())
    m3=json.loads((ROOT/'configs/member3.json').read_text())
    torch.set_num_threads(shared['threads'])
    torch.use_deterministic_algorithms(True)
    env=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform(),machine=platform.machine(),device='cpu',threads=torch.get_num_threads(),deterministic_algorithms=True)
    save_json(ROOT/'results/member3/environment.json',env)
    implementation_files=['common/data.py','common/models.py','common/train.py','members/optimizers.py','members/training.py']
    phasesrc=hashlib.sha256(b''.join((ROOT/'src'/p).read_bytes() for p in implementation_files)).hexdigest()
    for task,tc in shared['tasks'].items():
        if args.task and task!=args.task: continue
        data=load_task(task,shared,ROOT)
        if args.phase=='main':
            if m3['status']!='FROZEN_AFTER_VALIDATION_TUNING': raise RuntimeError('Run all tuning before main')
            seeds=shared['main_seeds']; steps=shared['main_updates']
        else:
            seeds=[shared['tuning_seed']]; steps=5 if args.phase=='smoke' else shared['tuning_updates']
        for seed in seeds:
            # Reverse the within-pair order on even main seeds to reduce a fixed order effect.
            names=m3['optimizers'] if seed%2 else list(reversed(m3['optimizers']))
            for name in names:
                rates=([m3['selected_learning_rates'][task][name]] if args.phase=='main'
                       else [0.001] if args.phase=='smoke' else shared['lr_candidates'])
                for lr in rates:
                    run_id=f'{task}__{name}__seed{seed}__lr{lr:g}'
                    out=ROOT/'results/member3/raw'/args.phase/run_id
                    summaryfile=out/'summary.json'
                    if summaryfile.exists():
                        old=json.loads(summaryfile.read_text())
                        if old.get('source_sha256')!=phasesrc: raise RuntimeError('Source changed; use a separate output directory')
                        print('EXISTS',run_id,old['status'],flush=True); continue
                    out.mkdir(parents=True,exist_ok=True)
                    model,init_hash=make_model(task,tc,seed,data.metadata)
                    opt=make_optimizer(name,model.parameters(),lr,shared['beta2'],shared['epsilon'])
                    record=dict(run_id=run_id,phase=args.phase,task=task,optimizer=name,seed=seed,learning_rate=lr,beta2=shared['beta2'],epsilon=shared['epsilon'],max_updates=steps,split_hash=data.split_hash,initialization_hash=init_hash,source_sha256=phasesrc,parameter_count=sum(p.numel() for p in model.parameters()),start_utc=datetime.now(timezone.utc).isoformat(),environment=env)
                    print('START',args.phase,run_id,flush=True)
                    try:
                        metrics,summary=train_one(model,opt,data,task,seed,steps,tc['batch_size'],shared['eval_interval'],tc['target_val_ce'],shared['eval_examples'],args.phase=='main',shared['eval_batch_size'])
                        with (out/'metrics.csv').open('w',newline='') as f:
                            writer=csv.DictWriter(f,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
                        record.update(summary,status='completed',test_evaluated=args.phase=='main')
                        # Final weights allow evaluation audits, not further model selection.
                        if args.phase=='main': torch.save(model.state_dict(),out/'final_model.pt')
                        print('DONE',run_id,'val',round(summary['final_val_ce'],5),'seconds',round(summary['elapsed_s'],2),flush=True)
                    except Exception as error:
                        record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
                        print('FAILED',run_id,repr(error),flush=True)
                    save_json(summaryfile,record)
    if args.phase=='tune' and not args.task:
        selected={}; records=[]
        for task in shared['tasks']:
            selected[task]={}
            for name in m3['optimizers']:
                candidates=[]
                for path in (ROOT/'results/member3/raw/tune').glob(f'{task}__{name}__*/summary.json'):
                    row=json.loads(path.read_text()); records.append(row)
                    if row['status'] in ('success','completed'): candidates.append(row)
                if len(candidates)!=len(shared['lr_candidates']): raise RuntimeError('All preregistered tuning trials must finish successfully before freezing')
                best=min(candidates,key=lambda r:(r['final_val_ce'],r['learning_rate']))
                selected[task][name]=best['learning_rate']
        m3.update(status='FROZEN_AFTER_VALIDATION_TUNING',selected_learning_rates=selected,selection_rule='minimum validation CE at 100 gradient calls, seed 11; tie -> smaller LR',frozen_utc=datetime.now(timezone.utc).isoformat())
        save_json(ROOT/'configs/member3.json',m3)
        print('FROZEN',selected,flush=True)

if __name__=='__main__': main()

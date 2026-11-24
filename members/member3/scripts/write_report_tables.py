"""Refresh the two Member 3 TeX tables from measured aggregate CSVs."""
import csv, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT/'results/member3/tables'/name).open() as f: return list(csv.DictReader(f))
def number(x,p):
    s=f'{float(x):.{p}f}'
    return s[1:] if s.startswith('0.') else s
def pm(row,mean,sd,p): return '$'+number(row[mean],p)+r'\pm'+number(row[sd],p)+'$'

def main():
    mainrows=read('main_results.csv');diags=read('diagnostic_results.csv')
    body=[]; tasknames=['Fashion','CIFAR','Text']
    for i,r in enumerate(mainrows):
        body.append(' & '.join([tasknames[i//2] if i%2==0 else '', 'RMSProp' if r['optimizer']=='rmsprop' else 'ADOPT-0',pm(r,'train_ce_mean','train_ce_sd',3),pm(r,'val_ce_mean','val_ce_sd',3),pm(r,'test_metric_mean','test_metric_sd',3 if r['task']=='tiny_shakespeare' else 2),pm(r,'elapsed_s_mean','elapsed_s_sd',2)])+r'\\')
    main_table='\n'.join(body)
    body=[]
    for method,label in [('rmsprop','RMSProp'),('adopt_no_momentum','ADOPT-0'),('current_moment_matched_clip','Current moment, matched clip'),('adopt_no_clip','ADOPT-0, no clip')]:
        rows=[next(r for r in diags if r['optimizer']==method and r['condition']==f'beta2={beta}' and r['metric_name']=='tail_mean_objective') for beta in [.9,.99,.999]]
        body.append(' & '.join([label]+[pm(r,'mean','sd',4) for r in rows])+r'\\')
    diag_table='\n'.join(body)
    p=ROOT/'report/sections/section4_experiments_member3.tex';text=p.read_text()
    tables=iter([main_table,diag_table])
    text=re.sub(r'(\\begin\{tabular\}.*?\\midrule\n).*?(\\bottomrule\\end\{tabular\})',lambda m:m.group(1)+next(tables)+'\n'+m.group(2),text,flags=re.S)
    p.write_text(text)
    print('Updated two TeX tables from main_results.csv and diagnostic_results.csv')

if __name__=='__main__': main()

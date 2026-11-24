import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from src.members.optimizers import RMSProp, ADOPT0

spec=importlib.util.spec_from_file_location('adopt_official',Path(__file__).resolve().parents[1]/'reference/adopt_official.py')
official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)

@pytest.mark.parametrize('beta',[.9,.999])
@pytest.mark.parametrize('kind',['rmsprop','adopt','adopt_unclipped'])
def test_matches_external_reference(beta,kind):
    a=torch.nn.Parameter(torch.tensor([1.,-2.,0.],dtype=torch.float64)); b=torch.nn.Parameter(a.detach().clone())
    if kind=='rmsprop':
        ours=RMSProp([a],lr=.003,beta2=beta,eps=1e-8)
        ref=torch.optim.RMSprop([b],lr=.003,alpha=beta,eps=1e-8,momentum=0,centered=False,foreach=False)
    else:
        ours=ADOPT0([a],lr=.003,beta2=beta,eps=1e-8,clip=kind=='adopt')
        ref=official.ADOPT([b],lr=.003,betas=(0.,beta),eps=1e-8,clip_lambda=(lambda step:step**.25) if kind=='adopt' else None,foreach=False)
    rng=torch.Generator().manual_seed(12)
    grads=[torch.zeros(3,dtype=torch.float64),torch.tensor([100.,1e-10,-3.],dtype=torch.float64)]
    grads += [torch.randn(3,generator=rng,dtype=torch.float64) for _ in range(30)]
    for g in grads:
        a.grad=g.clone(); b.grad=g.clone(); ours.step(); ref.step()
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('cls',[RMSProp,ADOPT0])
def test_resume_and_missing_gradient(cls):
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.float64)); opt=cls([p])
    opt.step(); assert len(opt.state)==0
    for g in [[1.,2.],[-3.,0.]]:
        p.grad=torch.tensor(g,dtype=torch.float64); opt.step()
    q=torch.nn.Parameter(p.detach().clone()); restored=cls([q]); restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for g in [[0.,2.],[-1.,8.]]:
        p.grad=torch.tensor(g,dtype=torch.float64);q.grad=p.grad.clone();opt.step();restored.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)

def test_adopt_first_gradient_initializes_only():
    p=torch.nn.Parameter(torch.tensor([2.,-3.])); o=ADOPT0([p]);p.grad=torch.tensor([0.,4.]);o.step()
    torch.testing.assert_close(p,torch.tensor([2.,-3.]));torch.testing.assert_close(o.state[p]['square_avg'],torch.tensor([0.,16.]))

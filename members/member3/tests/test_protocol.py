import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from src.common.data import TaskData
from src.common.train import train_one as original_train
from src.members.training import train_one as instrumented_train
from src.members.optimizers import make_optimizer

@pytest.mark.parametrize('variant',['rmsprop','adopt_no_momentum'])
def test_instrumentation_preserves_member4_training(variant):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    dataset=TensorDataset(torch.randn(16,3),torch.arange(16)%2)
    data=TaskData(dataset,dataset,dataset,'test',{})
    a=torch.nn.Linear(3,2); b=torch.nn.Linear(3,2); b.load_state_dict(a.state_dict())
    x,sa=original_train(a,make_optimizer(variant,a.parameters(),.001),data,'test',4,7,4,3,.8)
    y,sb=instrumented_train(b,make_optimizer(variant,b.parameters(),.001),data,'test',4,7,4,3,.8)
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    for ra,rb in zip(x,y):
        for key in ['step','train_ce','val_ce','val_accuracy_pct','learning_rate']: assert ra[key]==rb[key]
    for key in ['final_train_ce','final_val_ce','test_accuracy_pct','test_ce','target_step','optimizer_state_bytes']: assert sa[key]==sb[key]

@pytest.mark.parametrize('variant',['rmsprop','adopt_no_momentum','current_moment_matched_clip','adopt_no_clip'])
def test_numpy_diagnostic_matches_torch_optimizer(variant):
    path=Path(__file__).resolve().parents[1]/'scripts/run_diagnostic.py'
    spec=importlib.util.spec_from_file_location('diag',path); diag=importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
    noise=np.array([[-1.,9.,-1.,-1.,9.]])
    history=diag.trajectory(noise,variant,.9,.01)
    p=torch.nn.Parameter(torch.tensor([2.],dtype=torch.float64)); opt=make_optimizer(variant,[p],.01,.9)
    for t in range(noise.shape[1]):
        p.grad=p.detach().tanh()+noise[0,t];opt.step()
        np.testing.assert_allclose(p.detach().numpy(),history[:,t+1],rtol=1e-13,atol=1e-13)

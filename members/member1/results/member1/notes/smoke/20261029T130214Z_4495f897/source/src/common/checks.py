"""Offline unit checks for shared data/model/training behavior (fixture data only).

Real-data smoke tests are separate: scripts/run_member.py --member 1 --phase smoke.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .data import CharacterWindows, TaskData, load_task
from .models import make_model
from .train import PolynomialAverager, batch_indices, cross_entropy, evaluate, evaluation_mode, train_one
from .utils import hash_state
from src.members.member1_optimizers import SGD


class SharedChecks(unittest.TestCase):
    def test_models_shapes_and_seeded_initialization(self):
        configs = [dict(model="mlp", widths=[784, 128, 64, 10]),
                   dict(model="small_cnn", conv_channels=[16,32], kernel_size=3, conv_padding=1),
                   dict(model="char_gru", vocab_size=66, embedding_dim=32, hidden_size=64, layers=1)]
        inputs = [torch.zeros(2,1,28,28), torch.zeros(2,3,32,32), torch.zeros(2,64,dtype=torch.long)]
        shapes = [(2,10), (2,10), (2,64,66)]
        for cfg, x, expected in zip(configs, inputs, shapes):
            rng_state = torch.get_rng_state().clone()
            model, info = make_model(cfg, 11)
            self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
            again, repeated = make_model(cfg, 11)
            self.assertEqual(info["initialization_hash"], repeated["initialization_hash"])
            self.assertEqual(tuple(model(x).shape), expected)
            self.assertEqual(hash_state(model), hash_state(again))

    def test_text_splits_vocabulary_and_window_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "data/raw/tiny_shakespeare"
            raw.mkdir(parents=True)
            text = "abcd" * 20 + "Z" * 10 + "Y" * 10
            (raw / "input.txt").write_text(text)
            cfg = dict(repo_root=temp, task="tiny_shakespeare", train_count=4, validation_count=2,
                       train_eval_count=2, split_seed=6704, train_fraction=.8, validation_fraction=.1,
                       sequence_length=3, text_stride=3)
            a, b = load_task(cfg, 1), load_task(cfg, 999)
            self.assertEqual(a.metadata["split_hash"], b.metadata["split_hash"])
            manifest = json.loads((root / a.metadata["manifest"]).read_text())
            self.assertEqual(manifest["character_boundaries"], [0,80,90,100])
            self.assertNotIn("Z", manifest["vocabulary"])
            self.assertNotIn("Y", manifest["vocabulary"])
            self.assertTrue(torch.equal(a.validation[0][0], torch.zeros(3,dtype=torch.long)))
            windows = CharacterWindows("abcdefg", {c:i+1 for i,c in enumerate("abcdefg")},3,3)
            self.assertEqual(len(windows),2)
            self.assertEqual(windows[1][1].tolist(),[5,6,7])
            with self.assertRaises(IndexError):
                windows[2]

    def test_batch_order_and_epoch_coverage(self):
        a, b = batch_indices(10,4,11), batch_indices(10,4,11)
        batches = [next(a) for _ in range(6)]
        self.assertEqual(batches,[next(b) for _ in range(6)])
        self.assertEqual(sorted(sum(batches[:3],[])),list(range(10)))
        self.assertEqual(sorted(sum(batches[3:],[])),list(range(10)))
        self.assertEqual([len(x) for x in batches[:3]],[4,4,2])

    def test_weighted_image_and_character_ce(self):
        logits = torch.tensor([[4.,0.],[0.,2.],[0.,0.]])
        labels = torch.tensor([0,0,1])
        model = nn.Identity().train()
        expected = float(cross_entropy(logits, labels))
        result = evaluate(model,TensorDataset(logits,labels),2,False)
        self.assertAlmostEqual(result["ce"],expected,places=6)
        self.assertAlmostEqual(result["accuracy_pct"],100/3)
        self.assertTrue(model.training)
        text = evaluate(model,TensorDataset(logits.unsqueeze(0),labels.unsqueeze(0)),1,True)
        self.assertAlmostEqual(text["ce"],expected,places=6)
        self.assertIsNone(text["accuracy_pct"])

    def test_averager_formula_and_resume(self):
        base = nn.Linear(1,1,bias=False)
        avg = PolynomialAverager(base,8.)
        with torch.no_grad():
            base.weight.fill_(2.)
            avg.update(base)
            self.assertEqual(float(avg.model.weight.detach()[0,0]),2.)
            base.weight.fill_(4.)
            avg.update(base)  # weight = 9/10
            torch.testing.assert_close(avg.model.weight,torch.tensor([[3.8]]))
            restored = PolynomialAverager(base,0.)
            restored.load_state_dict(deepcopy(avg.state_dict()))
            base.weight.fill_(6.)
            avg.update(base)
            restored.update(base)
            torch.testing.assert_close(avg.model.weight,restored.model.weight,atol=0,rtol=0)
            self.assertEqual(float(base.weight[0,0]),6.)

    def test_optimizer_evaluation_hooks_restore_after_error(self):
        class Hooks:
            def __init__(self): self.calls=[]
            def eval(self): self.calls.append("eval")
            def train(self): self.calls.append("train")
        model, opt = nn.Linear(2,2).train(), Hooks()
        with self.assertRaisesRegex(RuntimeError,"fixture failure"):
            with evaluation_mode(model,opt):
                self.assertFalse(model.training)
                raise RuntimeError("fixture failure")
        self.assertTrue(model.training)
        self.assertEqual(opt.calls,["eval","train"])

    def test_trainer_evidence_failure_and_test_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            def make(nan=False):
                x=torch.ones(5,2)
                if nan: x[0,0]=float("nan")
                dataset=TensorDataset(x,torch.zeros(5,dtype=torch.long))
                def forbidden(): raise AssertionError("Test data accessed")
                data=TaskData(dataset,dataset,dataset,forbidden,dict(task="fashion_mnist",split_hash="fixture",
                               dataset_hash="fixture",train_ce_scope="fixed fixture set"))
                model=nn.Linear(2,2)
                cfg=dict(output_root=temp,member=1,phase="smoke",task="fashion_mnist",optimizer="sgd",seed=11,
                         max_updates=2,eval_interval=1,batch_size=2,eval_batch_size=3,evaluate_test=False,
                         initialization_hash=hash_state(model),model_hash="fixture",evaluation_point="raw")
                return model,SGD(model.parameters(),lr=.01),data,cfg
            model,opt,data,cfg=make()
            result=train_one(model,opt,data,cfg)
            self.assertEqual(result["status"],"completed")
            self.assertIsNone(result["test_ce"])
            self.assertEqual(result["last_evaluated_step"],2)
            model,opt,data,cfg=make(True)
            failure=train_one(model,opt,data,cfg)
            self.assertEqual(failure["status"],"diverged")
            self.assertTrue((Path(temp)/"raw"/failure["run_id"]/"summary.json").exists())
            model,opt,data,cfg=make()
            with self.assertRaises(ValueError):
                train_one(model,opt,data,dict(cfg,evaluate_test=True))
            with self.assertRaises(ValueError):
                train_one(model,opt,data,dict(cfg,phase="main",shared_status="DRAFT",member_status="DRAFT"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)

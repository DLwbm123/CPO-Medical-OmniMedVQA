import ast
from pathlib import Path
import unittest

from .grounding import grounded_advantages


class GCPilotTest(unittest.TestCase):
    def test_disabled_objective_three_step_bitwise_equivalence_on_cpu(self):
        import torch

        def run(with_identity_weight):
            parameter = torch.nn.Parameter(torch.tensor([.2,-.1,.3,-.4]))
            optimizer = torch.optim.SGD([parameter],lr=.01)
            trace = []
            for rewards in (torch.tensor([1.,0.,1.,0.]),torch.tensor([0.,1.,1.,0.]),
                            torch.tensor([1.,1.,0.,0.])):
                advantages = (rewards-rewards.mean())/rewards.std()
                batch = {'advantages':advantages.clone(),'completion_mask':torch.ones(4,2)}
                if with_identity_weight:
                    grounded_advantages(batch,[1.],[0.],4,0.,1.)
                logps = parameter.log_softmax(0)
                (-(logps*batch['advantages']).mean()+.04*logps.square().mean()).backward()
                trace.append((rewards.clone(),batch['advantages'].clone(),parameter.grad.clone()))
                optimizer.step()
                optimizer.zero_grad()
            return trace
        for control, ground in zip(run(False),run(True),strict=True):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(control,ground,strict=True)))

    def test_training_imports_exclude_gold_and_legacy_image_mask(self):
        root = Path(__file__).parent
        for name in ('gc_train.py','counterfactual.py','grounding.py'):
            tree = ast.parse((root/name).read_text())
            imported = {alias.name.split('.')[-1] for node in ast.walk(tree)
                        if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names}
            self.assertFalse(imported & {'load_labeled_eval','load_split','score_response',
                                         'mask_image_gate','precheck_grounding','confirm_eval'},name)

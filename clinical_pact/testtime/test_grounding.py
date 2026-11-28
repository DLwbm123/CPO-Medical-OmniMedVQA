from contextlib import contextmanager
import math
import unittest

from .grounding import blind_kl, evidence_weight, grounded_advantages, update_dual
from .precheck_grounding import choose_mechanism, summarize_evidence


class GroundingTest(unittest.TestCase):
    def test_weighting_must_follow_group_standardization(self):
        import torch

        rewards = torch.tensor([0.,1.,0.,1.])
        def normalized(values):
            return (values-values.mean())/values.std()
        advantages = normalized(rewards)
        batch = {'advantages':advantages.clone(),'completion_mask':torch.ones(4,3)}
        self.assertFalse(grounded_advantages(batch,[.25],[.5],4,.1,.9))
        self.assertTrue(torch.equal(batch['advantages'],advantages*.25))
        self.assertFalse(torch.equal(batch['advantages'],normalized(rewards*.25)))
        self.assertEqual(batch['completion_mask'].sum(),12)
        self.assertAlmostEqual(evidence_weight(math.log(.6),math.log(.4),1.5,.1)[1],.5)

    def test_m1_and_m2_exclusions_preserve_token_kl_mask(self):
        import torch

        batch = {'advantages':torch.ones(8),'completion_mask':torch.ones(8,4)}
        skip = grounded_advantages(batch,[.09,.8,.01,.2],[.5,.9,.1,.95],2,.1,.9)
        self.assertTrue(skip)
        self.assertEqual(batch['advantages'].count_nonzero(),0)
        self.assertEqual(batch['completion_mask'].sum(),32)

    def test_blind_kl_zero_reference_and_only_adapter_gradient(self):
        import torch

        base = torch.nn.Parameter(torch.tensor([1.,0.,-1.]),requires_grad=False)
        adapter = torch.nn.Parameter(torch.zeros(3))
        enabled = True
        @contextmanager
        def disable_adapter():
            nonlocal enabled
            enabled = False
            yield
            enabled = True
        def predict():
            return (base+adapter if enabled else base).log_softmax(0)
        initial = predict()
        with torch.no_grad(), disable_adapter():
            reference = predict()
        self.assertTrue(torch.equal(initial,reference))
        self.assertEqual(float(blind_kl(initial,reference).detach()),0.)
        with torch.no_grad():
            adapter[0] += .4
        blind_kl(predict(),reference).backward()
        self.assertIsNone(base.grad)
        self.assertFalse(reference.requires_grad)
        self.assertGreater(float(adapter.grad.abs().sum()),0.)

    def test_dual_and_p0_priority(self):
        cfg = {'blind_kl_budget_nats':.02,'dual_low_budget_fraction':.5,
               'dual_min':.1,'dual_max':100.,'dual_increase_factor':2.,'dual_decrease_factor':.5}
        self.assertEqual(update_dual(80,.03,cfg),100)
        self.assertEqual(update_dual(.15,.001,cfg),.1)
        self.assertEqual(update_dual(1,.01,cfg),1)
        p0 = {'choice_priority':['counterfactual','black'],'min_accuracy_gap':.05,'min_auroc':.6,
              'min_eligible_fraction':.2,'high_weight_threshold':.5,'grounding':{'min_weight':.1}}
        records = [{'evidence':1.,'weight':.7,'correct':True,'in_band':True},
                   {'evidence':0.,'weight':.05,'correct':False,'in_band':True}]
        success = summarize_evidence(records,p0)
        self.assertEqual(success['eligible_fraction'],.5)
        self.assertTrue(success['passed'])
        self.assertEqual(choose_mechanism({'Fundus':{'counterfactual':success,'black':success},
                         'Dermoscopy':{'counterfactual':{'passed':False},'black':success}},p0),'black')

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .guard import Guard, total_variation
from .votes import gather_votes, training_indices, vote_decision


def verify_rank_votes(rank, store):
    import torch.distributed as dist

    dist.init_process_group('gloo',init_method=store,rank=rank,world_size=2)
    try:
        local = ['A']*8 if rank == 0 else ['A']*2+['B']*2+[None]*4
        global_votes = gather_votes(local)
        guess = vote_decision(global_votes,'AB')
        chosen = training_indices(100,2,'joint-question')
        assert global_votes == ('A',)*10+('B',)*2+(None,)*4
        assert guess.pseudo_label == 'A' and guess.peak == 10 and guess.in_band
        assert len(set(chosen)) == 8 and len(guess.rewards([global_votes[i] for i in chosen])) == 8
    finally:
        dist.destroy_process_group()


class PilotRewardTest(unittest.TestCase):
    def test_two_rank_gather(self):
        from torch.multiprocessing import spawn

        with TemporaryDirectory() as temp:
            spawn(verify_rank_votes,args=((Path(temp)/'store').as_uri(),),nprocs=2)

    def test_majority_gate_rewards_and_subset(self):
        answers = ['A']*10 + ['B']*4 + [None]*2
        result = vote_decision(answers,'AB')
        self.assertEqual(result.pseudo_label,'A')
        self.assertTrue(result.in_band)
        chosen = training_indices(100,1,'case-1')
        self.assertEqual(chosen,training_indices(100,1,'case-1'))
        self.assertEqual(len(set(chosen)),8)
        self.assertEqual(result.rewards([answers[i] for i in chosen]),
                         [float(i < 10) for i in chosen])
        tied = vote_decision(['A']*8+['B']*8,'AB')
        self.assertIsNone(tied.pseudo_label)
        self.assertEqual(tied.rewards(['A','B',None]),[0.,0.,0.])
        self.assertFalse(vote_decision(['B']*15+[None],'AB').in_band)

    def test_guard_window_and_brake_stop(self):
        guard = Guard({'A':50,'B':50},window=2,consecutive_limit=2)
        self.assertEqual(guard.observe_update([{'A':8,'B':8}]),(0.,False,False))
        self.assertEqual(guard.observe_update([{'A':16},{'A':16}]),(.5,True,False))
        self.assertEqual(guard.observe_update([{'A':16}]),(.5,True,False))
        self.assertEqual(guard.observe_update([{'A':16}]),(.5,True,True))
        self.assertEqual(guard.observe_update([{'A':8,'B':8}]*2),(0.,False,False))
        self.assertIsNone(total_variation({}, {'A':1}))

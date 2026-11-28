import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .pilot_eval import evaluation_cohorts


class PilotCohortTest(unittest.TestCase):
    def test_deterministic_subsets_and_full_target_validation(self):
        spec = {'seed':100, 'task':'Fundus', 'validation_counts':{'CT':5, 'Fundus':5},
                'other_validation_n':3}
        rows = [(SimpleNamespace(id=str(i)), 'A') for i in range(5)]
        with patch('clinical_pact.testtime.pilot_eval.load_labeled_eval', return_value=rows) as loader:
            first = dict(evaluation_cohorts(Path('data'), spec))
            loader.return_value = rows[::-1]
            second = dict(evaluation_cohorts(Path('data'), spec))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {'CT','Fundus'})
        self.assertEqual(len(first['CT']),3)
        self.assertEqual(len(first['Fundus']),5)
        self.assertTrue(all(call.args[1]=='validation' for call in loader.call_args_list))

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .redact_train import redact
from .tt_data import load_unlabeled


class TrainingIsolationTest(unittest.TestCase):
    def test_training_modules_do_not_import_labeled_apis(self):
        root = Path(__file__).parent
        forbidden = {'load_labeled_eval', 'load_split', 'score_response', 'reward', 'redact_train'}
        for name in ('pilot_train.py','votes.py','guard.py','choices.py','tt_data.py'):
            tree = ast.parse((root/name).read_text())
            imports = {alias.name.split('.')[-1] for node in ast.walk(tree)
                       if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names}
            self.assertFalse(imports & forbidden,name)

    def test_only_question_and_whitelisted_metadata_are_transferred(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)/'raw'
            raw = source/'jsons'/'train'
            raw.mkdir(parents=True)
            (source/'images').mkdir()
            (source/'images'/'sample.png').touch()
            record = {'id':'synthetic','image':'sample.png','source_dataset':'synthetic',
                      'original_answer':'private-answer-fixture',
                      'conversations':[{'from':'human','value':'Question\nA. One\nB. Two'},
                                       {'from':'gpt','value':'private-answer-fixture'}]}
            (raw/'data.json').write_text(json.dumps([record]))
            dest = Path(temp)/'redacted'
            self.assertEqual(redact(source,dest,100),1)
            self.assertNotIn('private-answer-fixture',(dest/'unlabeled/train/data.json').read_text())
            items = load_unlabeled(dest,'train',100)
            self.assertFalse(hasattr(items[0],'answer'))

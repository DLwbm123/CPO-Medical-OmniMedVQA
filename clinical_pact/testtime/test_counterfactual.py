from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .counterfactual import CounterfactualPool, ImageRecord, image_record


class MetadataOnly:
    def __init__(self, values):
        self.values = values

    def __getitem__(self,key):
        if key not in ('id','image','source_dataset'):
            raise AssertionError('image selection must not read labels')
        return self.values[key]


class CounterfactualTest(unittest.TestCase):
    def test_same_source_reproducible_and_protected_images_excluded(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            names = ('real','same','other','holdout','validation','test','xray_holdout')
            for name in names:
                (root/name).touch()
            (root/'alias').symlink_to(root/'holdout')
            rows = []
            for name in (*names,'alias'):
                metadata = MetadataOnly({'id':name,'image':name,'source_dataset':'s' if name!='other' else 't'})
                rows.append(image_record(metadata,root))
            protected = [root/name for name in names[3:]]
            original = rows[0]
            pool = CounterfactualPool(rows,protected,100)
            shuffled = CounterfactualPool(list(reversed(rows)),protected,100)
            selected, fallback = pool.select(original)
            self.assertEqual(shuffled.select(original),(selected,fallback))
            self.assertEqual(selected.image,root/'same')
            self.assertFalse(fallback)
            self.assertTrue(all(r.image not in protected for r in pool.all_images))
            label_free_query = ImageRecord('new',root/'other','t')
            fallback_image, used_fallback = pool.select(label_free_query)
            self.assertTrue(used_fallback)
            self.assertNotEqual(fallback_image.image,label_free_query.image)

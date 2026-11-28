"""Select deterministic, label-free same-source image substitutions."""
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    id: str
    image: Path
    source_dataset: str


def image_record(row, images):
    path = (images / row['image']).resolve(strict=True)
    if not path.is_relative_to(images):
        raise ValueError('image escaped the task root')
    return ImageRecord(str(row['id']),path,row['source_dataset'])


class CounterfactualPool:
    def __init__(self, items, forbidden, seed):
        self.forbidden = frozenset(Path(p).resolve() for p in forbidden)
        self.seed = seed
        allowed = {}
        for row in sorted(items,key=lambda i: i.id):
            path = Path(row.image).resolve(strict=True)
            if path not in self.forbidden:
                allowed.setdefault((row.source_dataset,path),ImageRecord(row.id,path,row.source_dataset))
        self.all_images = tuple(allowed.values())
        self.by_source = {}
        for item in self.all_images:
            self.by_source.setdefault(item.source_dataset,[]).append(item)
        if len({r.image for r in self.all_images}) < 2:
            raise ValueError('fewer than two allowed training images in the modality')

    def select(self, item):
        source = Path(item.image).resolve(strict=True)
        candidates = [r for r in self.by_source.get(item.source_dataset,()) if r.image != source]
        fallback = not candidates
        if fallback:
            candidates = [r for r in self.all_images if r.image != source]
        value = blake2b(f'{self.seed}:{item.id}'.encode(),digest_size=8).digest()
        result = candidates[int.from_bytes(value,'big') % len(candidates)]
        if result.image in self.forbidden or result.image == source:
            raise ValueError('forbidden counterfactual image')
        return result,fallback

"""Lazy DIPS-Plus Pair loading; no docking augmentation or dataset conversion."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

NAMES = 'TYR ARG PHE GLY ILE VAL ALA TRP GLU HIS CYS ASN MET ASP THR SER LYS LEU GLN PRO'.split()
ALIASES = {'HIP':'HIS', 'HIE':'HIS', 'HID':'HIS', 'CYX':'CYS', 'CYM':'CYS', 'SEP':'SER', 'TPO':'THR', 'PTR':'TYR',
           'LEV':'LEU', 'MEU':'MET', 'GLV':'GLU', 'CYT':'CYS', 'HIZ':'HIS',
           'GLM':'GLU', 'ASQ':'ASP', 'TYS':'TYR', 'GLZ':'GLY'}

def ca_tensors(frame):
    required = {'atom_name', 'resname', 'x', 'y', 'z'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    ca = frame.loc[frame.atom_name.astype(str).str.strip() == 'CA'].copy()
    keys = [c for c in ('model', 'chain', 'residue') if c in ca.columns]
    if not keys or ca.duplicated(keys).any():
        raise ValueError('Missing residue identifiers or duplicate CA atoms/models; resolve before training')
    xyz = ca[['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=True)
    if len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError('Need at least two finite CA coordinates per protein')
    types = [NAMES.index(n) if n in NAMES else 20 for n in
             (ALIASES.get(str(v).strip(), str(v).strip()) for v in ca.resname)]
    return torch.from_numpy(xyz), torch.tensor(types, dtype=torch.long)

def read_pair(path):
    # Load only trusted dataset files. Legacy dill Pair objects may require atom3.
    import dill
    with open(path, 'rb') as handle:
        pair = dill.load(handle)
    if isinstance(pair, dict):
        left, right = pair['df0'], pair['df1']
    else:
        left, right = pair.df0, pair.df1
    return (*ca_tensors(left), *ca_tensors(right))

def resolve_manifest(root, split, explicit=None):
    root = Path(root).resolve()
    if explicit:
        manifest = Path(explicit).resolve()
    else:
        candidates = []
        for name in (f'pairs-postprocessed-{split}.txt', f'{split}.txt'):
            candidates = list(root.rglob(name))
            if candidates:
                break
        if len(candidates) != 1:
            raise ValueError(f'Expected one {split} manifest below {root}, found {len(candidates)}; supply --{split}-list')
        manifest = candidates[0]
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return manifest

def check_disjoint(*datasets):
    seen = set()
    for dataset in datasets:
        paths = {p.resolve() for p in dataset.paths}
        if seen & paths:
            raise ValueError('Split manifests overlap')
        seen.update(paths)

class DIPSPlus(Dataset):
    def __init__(self, root, manifest, max_pairs=1000000):
        self.root = Path(root)
        self.files = [line.strip() for line in Path(manifest).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith('#')]
        if not self.files:
            raise ValueError('Empty split manifest')
        self.max_pairs = max_pairs
        self.paths = []
        for name in self.files:
            options = [self.root / name, Path(manifest).parent / name]
            path = next((p for p in options if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f'{name}: not found relative to data root or manifest directory')
            self.paths.append(path.resolve())
        if len(set(self.paths)) != len(self.paths):
            raise ValueError('Duplicate samples in manifest')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            item = read_pair(self.paths[index])
        except Exception as exc:
            raise RuntimeError(f'Failed loading {self.paths[index]}: {exc}') from exc
        if len(item[0]) * len(item[2]) > self.max_pairs:
            raise ValueError(f'{self.files[index]} exceeds --max-pairs; curate the manifest or raise limit')
        return item

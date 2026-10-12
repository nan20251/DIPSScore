"""Data and train/resume/evaluation lifecycle tests with a lightweight stand-in.

These verify pipeline wiring, not the unavailable geometric encoder.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import dill
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import DIPSPlus, resolve_manifest, check_disjoint
from metrics import contact_metrics

class TinyModel(torch.nn.Module):
    def __init__(self, device='cpu', topk=1500):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., device=device))
        self.topk = topk
        self.network = types.SimpleNamespace(distance=types.SimpleNamespace(mdn_loss_fn=self.nll))

    def nll(self, logpi, sigma, mu, distances):
        return -torch.logsumexp(logpi + torch.distributions.Normal(mu, sigma).log_prob(distances), 1)

    def distributions(self, left, lt, right, rt):
        count = len(left)*len(right)
        mu = self.weight.expand(count, 1)
        return (torch.zeros_like(mu), torch.ones_like(mu), mu), self.weight.expand(count)

    def training_loss(self, *item):
        loss = self.weight.square()
        return loss, loss.detach(), loss.detach()*0

class PipelineTests(unittest.TestCase):
    def test_metrics_ties_and_no_positives(self):
        m = contact_metrics(torch.zeros(2), torch.tensor([1,0]), 2)
        self.assertEqual(m['contact_auroc'], .5)
        self.assertEqual(m['contact_ap'], .5)
        self.assertIsNone(contact_metrics(torch.ones(2), torch.zeros(2), 1)['contact_ap'])

    def test_data_train_resume_evaluate(self):
        import train
        import evaluate
        # Initialize lazy optimizer imports before patch.dict restores sys.modules.
        torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.))])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, split in enumerate(['train', 'val', 'test']):
                frame = pd.DataFrame(dict(atom_name=['CA','CA'],resname=['ALA','TYR'],
                    chain=['A','A'],residue=['1','2'],x=[0.,1.],y=[0.,0.],z=[0.,0.]))
                with (root/f'{split}.dill').open('wb') as handle:
                    dill.dump(types.SimpleNamespace(df0=frame, df1=frame.assign(z=2.)), handle)
                (root/f'pairs-postprocessed-{split}.txt').write_text(f'{split}.dill\n')
            datasets = [DIPSPlus(root, resolve_manifest(root, split)) for split in ['train','val','test']]
            check_disjoint(*datasets)
            self.assertEqual(datasets[0][0][1].tolist(), [6,0])
            with self.assertRaises(ValueError):
                check_disjoint(datasets[0],datasets[0])
            module = types.ModuleType('dips_score'); module.DIPSScore = TinyModel
            out = root/'run'
            argv = ['train.py','--data-root',str(root),'--output',str(out),'--epochs','1','--device','cpu']
            with patch.dict(sys.modules, {'dips_score':module}), patch.object(sys,'argv',argv):
                train.main()
            self.assertTrue((out/'best.pt').is_file())
            with patch.dict(sys.modules, {'dips_score':module}), patch.object(sys,'argv',argv[:-4]+['--epochs','2','--device','cpu','--resume',str(out/'last.pt')]):
                train.main()
            self.assertEqual(len((out/'history.jsonl').read_text().splitlines()),2)
            args = ['evaluate.py','--data-root',str(root),'--checkpoint',str(out/'best.pt'),
                    '--split','test','--device','cpu','--output',str(out/'test')]
            with patch.dict(sys.modules, {'dips_score':module}), patch.object(sys,'argv',args):
                evaluate.main()
            summary = json.loads((out/'test/metrics.json').read_text())
            self.assertEqual(summary['samples'],1)
            self.assertTrue((out/'test/per_complex.csv').is_file())

if __name__ == '__main__':
    unittest.main()

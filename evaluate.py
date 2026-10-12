"""Evaluate native distance likelihood and residue contact prediction."""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import DIPSPlus, resolve_manifest
from checkpoint import load_checkpoint
from metrics import contact_metrics

def evaluate(model, dataset, workers=0):
    model.eval()
    rows = []
    with torch.no_grad():
        for index, item in enumerate(DataLoader(dataset, batch_size=None, num_workers=workers)):
            params, logits = model.distributions(*item)
            distances = torch.cdist(item[0].to(logits.device), item[2].to(logits.device)).flatten()
            selected = distances < 100.
            if not selected.any():
                raise ValueError(f'No native pairs within 100 A: {dataset.files[index]}')
            nll = model.network.distance.mdn_loss_fn(
                *(p[selected] for p in params), distances[selected, None]).mean()
            labels = (distances < 15.).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            k = min(int(2.5*(len(item[0])+len(item[2]))), model.topk, logits.numel())
            idx = logits.topk(k).indices
            score = model.network.distance.mdn_loss_fn(*(p[idx] for p in params), distances[idx, None]).mean()
            row = dict(sample=dataset.files[index], nll=nll.item(), contact_bce=bce.item(),
                       total_loss=(nll+bce).item(), score=score.item(),
                       **contact_metrics(logits, labels, k))
            if not all(torch.isfinite(torch.tensor(row[key])) for key in ('nll','contact_bce','score')):
                raise FloatingPointError(f'Nonfinite evaluation: {row["sample"]}')
            rows.append(row)
    keys = ('nll','contact_bce','total_loss','score','contact_ap','contact_auroc','precision_at_k','recall_at_k')
    summary = {'samples': len(rows), 'aggregation': 'macro average over complexes'}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = sum(values)/len(values) if values else None
        summary[key+'_valid_samples'] = len(values)
    return summary, rows

def save_results(directory, summary, rows):
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (out/'per_complex.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--val-list'); p.add_argument('--test-list')
    p.add_argument('--output', default='runs/dipsscore/evaluation')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max-pairs', type=int, default=1000000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    from dips_score import DIPSScore
    checkpoint = load_checkpoint(a.checkpoint, a.device)
    model = DIPSScore(a.device, checkpoint.get('config', {}).get('topk',1500))
    model.load_state_dict(checkpoint['model'])
    manifest = resolve_manifest(a.data_root, a.split, getattr(a, a.split+'_list'))
    dataset = DIPSPlus(a.data_root, manifest, a.max_pairs)
    if a.split == 'test':
        used = set(checkpoint.get('train_paths', []) + checkpoint.get('val_paths', []))
        if used & {str(path) for path in dataset.paths}:
            raise ValueError('Test data overlaps training/validation paths saved in checkpoint')
    summary, rows = evaluate(model, dataset, a.workers)
    summary.update(split=a.split, checkpoint=str(Path(a.checkpoint).resolve()), manifest=str(manifest))
    save_results(a.output, summary, rows)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

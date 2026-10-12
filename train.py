"""Native-distance likelihood training on official DIPS-Plus train/validation lists."""
import argparse
import random
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import DIPSPlus, resolve_manifest, check_disjoint
from checkpoint import load_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--train-list')
    p.add_argument('--val-list')
    p.add_argument('--output', default='runs/dipsscore')
    p.add_argument('--resume', help='Resume from last.pt; epochs is the total target epoch count')
    p.add_argument('--topk', type=int, default=1500)
    p.add_argument('--patience', type=int, default=10, help='Early stopping epochs; 0 disables')
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max-pairs', type=int, default=1000000)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.epochs < 1 or a.accumulation_steps < 1 or a.patience < 0 or a.lr <= 0:
        p.error('Invalid epochs, accumulation steps, patience or learning rate')
    from dips_score import DIPSScore
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train = DIPSPlus(a.data_root, resolve_manifest(a.data_root, 'train', a.train_list), a.max_pairs)
    val = DIPSPlus(a.data_root, resolve_manifest(a.data_root, 'val', a.val_list), a.max_pairs)
    check_disjoint(train, val)
    # One complex per step avoids padding O(B * max(N_l) * max(N_r)).
    loaders = [DataLoader(ds, batch_size=None, shuffle=(i == 0), num_workers=a.workers)
               for i, ds in enumerate((train, val))]
    model = DIPSScore(a.device, a.topk)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    start, stale = 0, 0
    if a.resume:
        saved = load_checkpoint(a.resume, a.device)
        for key, dataset in [('train_paths',train), ('val_paths',val)]:
            if saved.get(key) != [str(p) for p in dataset.paths]:
                raise ValueError('Resume requires unchanged data manifests/order')
        if saved['config']['topk'] != a.topk:
            raise ValueError('Resume requires unchanged topk')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best, stale = saved['epoch'], saved['best'], saved['stale']
        random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'].cpu())
        if torch.cuda.is_available() and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    (out/'config.json').write_text(json.dumps(vars(a), indent=2)+'\n')
    print(f'DIPSScore: train={len(train)}, val={len(val)}, device={a.device}', flush=True)
    for epoch in range(start, a.epochs):
        metrics = []
        for i, loader in enumerate(loaders):
            model.train(i == 0)
            totals = np.zeros(3)
            with torch.set_grad_enabled(i == 0):
                optimizer.zero_grad(set_to_none=True)
                for step, item in enumerate(loader):
                    loss, nll, cp = model.training_loss(*item)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training/validation loss')
                    if i == 0:
                        group_start = (step // a.accumulation_steps) * a.accumulation_steps
                        group_size = min(a.accumulation_steps, len(loader)-group_start)
                        (loss/group_size).backward()
                        if (step+1) % a.accumulation_steps == 0 or step+1 == len(loader):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    totals += [loss.item(), nll.item(), cp.item()]
            metrics.append((totals / len(loader)).tolist())
        improved = metrics[1][0] < best
        if improved:
            best = metrics[1][0]
            stale = 0
        else:
            stale += 1
        checkpoint = dict(model_name='DIPSScore', model=model.state_dict(), epoch=epoch + 1,
                          optimizer=optimizer.state_dict(), config=vars(a), metrics=metrics,
                          best=best, stale=stale,
                          train_paths=[str(p) for p in train.paths], val_paths=[str(p) for p in val.paths],
                          rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                   torch=torch.get_rng_state(),
                                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        torch.save(checkpoint, out / 'last.pt')
        if improved:
            torch.save(checkpoint, out / 'best.pt')
        with (out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(epoch=epoch+1, train=metrics[0], val=metrics[1], best=best))+'\n')
        print(f'epoch={epoch+1} train(total,nll,cp)={metrics[0]} val={metrics[1]}', flush=True)
        if a.patience and stale >= a.patience:
            print('Early stopping', flush=True)
            break

if __name__ == '__main__':
    main()

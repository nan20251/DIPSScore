# DIPSScore

Model class: `DIPSScore`, defined in `dips_score.py`. This named model uses
the EBMDock-derived architecture described below; renaming does not introduce
a new architecture. Train your own weights on DIPS-Plus.

## Direct training and evaluation

Run from this directory in an environment with the dependencies below installed:

```bash
python train.py --data-root /data/DIPS/final/raw
python evaluate.py --data-root /data/DIPS/final/raw --checkpoint runs/dipsscore/best.pt
```

CUDA is selected automatically when available. Official train/val manifests
are found recursively under data-root; train.txt/val.txt are also accepted.
File paths resolve relative to data-root or the manifest's directory. Ambiguous
manifests require explicit --train-list/--val-list arguments. Data is read lazily.

Training outputs config.json, history.jsonl, best.pt and last.pt. Best is selected
by validation NLL+BCE. Early stopping defaults to 10 epochs without improvement.
--accumulation-steps 4 averages gradients over four complexes, including a
correctly normalized final partial group. No padded multi-complex batching.

```bash
python train.py --data-root /data/DIPS/final/raw --resume runs/dipsscore/last.pt --epochs 100
python evaluate.py --data-root /data/DIPS/final/raw --checkpoint runs/dipsscore/best.pt \
  --split test --test-list /data/heldout/test.txt --output runs/dipsscore/test
```

Resume restores model, optimizer, best metric, patience and random generator
states. Keep manifests, topk, optimization settings and worker configuration
unchanged for reproducible continuation. The output directory should be the
original run directory. --epochs is the total target, not additional epochs.

Evaluation writes metrics.json and per_complex.csv: native-distance NLL,
contact BCE, total loss, top-K MDN score, contact AP/AUROC and precision/recall
at the selected K. Metrics are macro averages over complexes; AP/AUROC/recall
are null when undefined and valid sample counts are included. Contact truth
is CA distance <15 A, not the dataset's pos_idx annotations.
These metrics evaluate distance/contact learning; they do not measure docking
decoy ranking or affinity prediction. Validation is the default, not a held-out
test claim. Test requires your independent manifest; exact saved training and
validation path overlap is rejected. Homology leakage requires separate curation.

Pipeline tests: `python -m unittest discover -s tests -v`. A lightweight stand-in
checks data, checkpoint/resume and evaluation wiring, not encoder correctness.

Independent train/score entry points; no import of the parent EBMDock project.
The encoder/vendor code is copied unchanged from the upstream EBMDock source.
Origin: https://github.com/wuhuaijin/EBMDock (ICLR 2024). The local source has
no root LICENSE; this extraction does not assert a new license for vendor code.
See [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for the source revision and attribution.

Dataset: https://github.com/BioinfoMachineLearning/DIPS-Plus
Use the official final/raw dill Pair files and corresponding
pairs-postprocessed-train.txt and pairs-postprocessed-val.txt manifests.
--data-root must be the directory against which manifest paths resolve.
No large dataset is bundled or automatically downloaded. HDF5 is not supported
by this first adapter. Only load trusted dill/checkpoint files.

Dependencies: start with upstream PyTorch 1.12.1/CUDA 11.6/PyG environment,
then install numpy, pandas, scipy, dill, biopandas and a compatible e3nn.
torch_cluster, torch_scatter and torch_geometric must match PyTorch/CUDA.
Legacy Pair unpickling may require atom3 from the DIPS-Plus environment.
The original checkout does not pin e3nn; compatibility needs an actual smoke run.

```bash
python train.py --data-root /data/DIPS/final/raw \
  --train-list /data/DIPS/final/raw/pairs-postprocessed-train.txt \
  --val-list /data/DIPS/final/raw/pairs-postprocessed-val.txt \
  --device cuda --output runs/dipsplus

python score.py --checkpoint runs/dipsplus/best.pt \
  --receptor receptor.pdb --ligand posed_ligand.pdb
```

Training is from scratch: shared encoder with contact and six-Gaussian MDN heads.
Loss = mean native-distance NLL for CA pairs below 100 A + binary contact BCE
(CA distance below 15 A). These labels intentionally follow EBMDock geometry,
not DIPS-Plus pos_idx's interface definition. Extra DIPS-Plus annotations are
not used. Both protein coordinates must retain their native relative placement.
CA-only and all-atom Pair tables are accepted; CA identities must be unique.
Protein graph cutoff is 8 A; residues use the upstream 21-category mapping.
No SGLD, negative-pose sampling, affinity or DockQ labels are used.

Inference selects top min(2.5*(Nl+Nr),1500,Nl*Nr) predicted-contact pairs and
returns mean MDN NLL, lower better. The contact model is jointly trained rather
than the original separately pretrained contact checkpoint. This is an
EBMDock-derived likelihood baseline, not reproduction of its CD/EBM training.
Validation selects by total native NLL+BCE, not decoy ranking performance.
Evaluate decoy ranking independently before treating it as a docking scorer.

Use official splits; do not randomly redistribute individual chain pairs.
Exact path overlap is checked, but homolog/structure leakage is not inferred.
The full pair feature matrix remains O(Nl*Nr); --max-pairs fails explicitly for
oversized samples. Curate split manifests consistently rather than silently
skipping samples. One complex per step; multi-GPU is not supported yet.

Real DIPS-Plus/encoder training remains unverified until the dataset and
compatible geometric dependencies are present. Test fixtures do not substitute
for a real dataset smoke run.

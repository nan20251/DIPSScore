# Source provenance

The `vendor/` files were copied without modification from the local EBMDock
checkout at revision `6726358e118b1966d57877392684220d43271df0`.
Upstream: https://github.com/wuhuaijin/EBMDock

Original work: Wu et al., “EBMDock: Neural Probabilistic Protein-Protein Docking
via a Differentiable Energy-Based Model”, ICLR 2024.

The wrapper, DIPS-Plus loader, training/evaluation entry points and pipeline
tests implement a jointly trained native-distance likelihood/contact baseline.
This is not a reproduction of upstream contrastive-divergence EBM training.

The local upstream checkout has no root LICENSE. No blanket license is asserted
for this repository. Existing source notices are preserved; individual vendor
files also include third-party references and notices.

DIPS-Plus data is not included. Obtain data and follow its terms from:
https://github.com/BioinfoMachineLearning/DIPS-Plus

No trained checkpoint is included. Syntax/CLI checks and lightweight pipeline
tests have passed; the geometric encoder has not been trained or validated on
real DIPS-Plus data in this environment.

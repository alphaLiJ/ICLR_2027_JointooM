# Why this repository is organized this way

The [WarpDrive](https://github.com/salesforce/warp-drive) repository separates
its CUDA/Python framework from environments, tutorials and tests, while its
README gives a small training example and build checks. The
[GPUDrive](https://github.com/Emerge-Lab/gpudrive) repository separates
simulator code, baselines, examples, data preparation and tests; its README
distinguishes installation, a first use, training integrations and dataset
acquisition. [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) keeps the
package, runnable scripts, documentation and tests visible at the top level.

JointLoom follows the same reviewer path: locate a claimed mechanism, run a
short semantic check, then locate the exact experiment path. Its navigation
is **representation first** because that is the paper's contribution:

1. `src/cuda_backend` owns compact joint state, transition kernels, cache
   updates and graph/token materialization.
2. `src/expert` and `src/mapf_cuda/training` own the CPU expert producer,
   complete-stage admission, ring lifetime and GPU learner consumption.
3. `src/baselines` and the baseline engines live beside the proposed path so
   protocol-matched comparisons can be inspected.
4. `experiments` records how claims were measured; `tests` checks semantic
   and lifecycle contracts. The mapping from each paper section to actual
   files is in `PAPER_TO_CODE.md`.

The legacy `expert.*` paths are intentionally retained. Moving hundreds of
imports solely to make the tree look cleaner would change a validated
experiment surface without improving reviewability. The public README and
paper-to-code map provide stable entry points for readers.

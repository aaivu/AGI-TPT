# ETF-TPT: Equiangular Tight Frame Test-Time Prompt Tuning

> **Calibration-aware Test-Time Prompt Tuning for Vision–Language Models**

Vision–language models (VLMs) such as CLIP achieve strong zero-shot
generalization by aligning image and text in a shared embedding space.
Test-Time Prompt Tuning (TPT) further improves zero-shot accuracy by
optimizing learnable prompt tokens per test image through entropy
minimization, but systematically miscalibrates the model: predicted
probabilities become persistently overconfident and no longer reflect the true
likelihood of correctness — an unacceptable failure mode in safety-critical
deployments. Existing calibration-aware variants disperse class text features
geometrically, yet they suffer from pairwise feature collapse, geometrically
infeasible targets, or unstable greedy gradients.

We propose **ETF-TPT**, which regularizes the per-test-image prompt geometry
toward an **Equiangular Tight Frame (ETF)** — the maximally uniform arrangement
of *N* unit vectors in *R^D* permitted by frame theory. We show that the ETF
target subsumes prior centroid-, orthogonality-, and angular-margin-based
regularizers as special or degenerate cases. While direct ETF regularization
sharply improves calibration, it homogenizes pre-trained semantic relations and
degrades accuracy. To resolve this tension we introduce **Semantic-Aware ETF
(LS-ETF)**, a margin-constrained, ReLU-gated relative target that enforces
near-uniform angular dispersion while preserving the semantic topology of the
underlying text encoder. The resulting regularizer is bounded below by the
Welch limit, adds no inference-time cost, and exposes only two stable
hyperparameters. Across 11 classification benchmarks and two CLIP backbones
(ViT-B/16 and RN50), **S-ETF-TPT** attains state-of-the-art calibration
(average ECE of **1.72** on ViT-B/16 vs. **2.61** for the strongest prior
method) while matching or improving zero-shot accuracy.

## 📥 Installation
```bash
#Steps to set up the environment
 1. git clone https://github.com/aaivu/ETF-TPT.git
 2. cd ETF-TPT
 3. conda env create -f environment.yml
 4. conda activate ctpt
```

## 📂 Datasets
We have conducted main experiments on fine-grained and natural distribution shift datasets:

- **Fine-grained datasets**:  
  1. ImageNet  
  2. Flower102  
  3. OxfordPets  
  4. SUN397  
  5. DTD  
  6. Food101  
  7. StanfordCars  
  8. Aircraft  
  9. UCF101  
  10. EuroSAT  
  11. Caltech101  

- **Natural distribution shift datasets**:  
  1. ImageNet-V2  
  2. ImageNet-A  
  3. ImageNet-R  
  4. ImageNet-Sketch

Follow this repository for datasets preparation: [TPT](https://github.com/azshue/TPT) 

## 🔧 Run Experiments
In each `.sh` file, you can edit the root dataset directory location (`data_root`) as well as configure the baseline, whether it's `RN50` or `ViT-B/16`. Also, you can switch between different experiment modes by changing `run_type`, whether it is baseline, tpt baselines, or calibration with the ETF regularizer (`tpt_ctpt`). For the ETF runs, `lambda_term` controls the regularization strength.

🏁 **Baseline Experiment**

```bash
#bash scripts/test_baseline.sh /I/DTD/Flower102/Food101/Cars/SUN397/Aircraft/Pets/Caltech101/UCF101/eurosat for fine-grained classification
  bash scripts/test_baseline.sh {dataset}
```

🎯 **TPT Experiment**
```bash
#Fine-grained classification
bash scripts/test_tpt_fg.sh {dataset}

#Natural distribution shift
bash scripts/test_tpt_ds.sh {dataset}
```

🔥 **ETF-TPT Experiment**

```bash
#Fine-grained classification
bash scripts/test_tpt_ctpt_fg.sh {dataset}

#Natural distribution shift
bash scripts/test_tpt_ctpt_ds.sh {dataset}
```

## 📄 Paper
The NeurIPS submission describing this work is available under [`writing/NeurIPs submitted ETF-TPT.pdf`](writing/NeurIPs%20submitted%20ETF-TPT.pdf).

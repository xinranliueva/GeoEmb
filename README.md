# Multimodal Spatial Representation Learning 

This repository contains code to:

1. Generate a synthetic multimodal geospatial dataset  
2. Train a shared masked graph autoencoder using self-supervised learning  
3. Evaluate learned spatial embeddings on downstream regression tasks  

The full pipeline is fully reproducible using the provided environment specification.

---

# Overview

The goal is to learn spatial embeddings from multimodal environmental data (wind and pollution) defined over a regional graph. The learned embeddings are evaluated on downstream spatial interpolation tasks.

Pipeline stages:

- Dataset generation  
- Self-supervised embedding pretraining  
- Downstream evaluation  

---

# Environment Setup

First, clone the repository:

```bash
git clone https://github.com/xinranliueva/GeoEmb.git
cd GeoEmb/
```

We provide an environment file for reproducibility.

## Create the environment

First, install **mamba** in the base environment (only needed once):

```bash
conda activate base
conda install -c conda-forge mamba
```

Then create the environment from the YAML file:

```bash
mamba env create -f environment.yml
```

Activate the environment:

```bash
conda activate geo
```

The environment name `geo` is specified inside `environment.yml`. It is currently set to `geo`, but feel free to change the name according to your preference.

You can modify the name by opening `environment.yml` and looking at the first line:

```yaml
name: geo
```

### Notes

- `mamba` makes solving the environment much faster and more reliable than `conda`.
- It works best with **conda 24 or newer**, and may not work as well with older versions.
- You only need to install `mamba` once.

---

## Verify installation

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Hardware

All experiments were run on:

- GPU: NVIDIA RTX A6000 
- OS: Linux  
- Python: 3.11
- Framework: PyTorch  

GPU is recommended but not required.

The code automatically uses GPU if available.

To explicitly select GPU:

```bash
python pretrain_shared.py --cuda 1
```

---

# Step 1: Generate Dataset

Run:

```bash
python data_generator.py --out_dir data --level postal
```

This will create:

```
data/region_graph_with_features_and_targets.npz
```

This step generates:

- Spatial graph structure  
- Wind features  
- Air quality features  
- Regression targets  

Runtime: less than 1 minute

---

# Step 2: Train Embedding Model

Run:

```bash
python pretrain_shared.py --cuda 1
```

This will create:

```
checkpoints/shared_final_emb_128.pt
```

This file contains the learned spatial embeddings.

Typical runtime on RTX A6000:

30–60 minutes  

---

# Step 3: Evaluate Embeddings

Run:

```bash
python eval.py --target env 

```

This will create:

```
results.csv
```

Metrics reported:

- MAE  
- RMSE  
- R²  

Methods evaluated:

- kNN  
- IDW  
- Embedding+MLP  

We pair embeddings with a standardized downstream predictor (scikit-learn MLPRegressor). This choice reflects a general downstream setting, including scenarios without GPU access, and ensures a reproducible comparison.

---

# Full Reproducibility Pipeline

Run the following commands in order:

```bash
conda env create -f environment.yaml
conda activate <env_name>

python data_generator.py --out_dir data

python pretrain_shared.py \
  --data data/region_graph_with_features_and_targets.npz \
  --out checkpoints

python eval.py \
  --input data/region_graph_with_features_and_targets.npz \
  --emb checkpoints/shared_final_emb_128.pt
```

---

# Project Structure

```
.
README.md
environment.yaml

data_generator.py
pretrain_shared.py
eval.py

models/
utils/
dataloader/
regressors/

data/
checkpoints/
results.csv
```

---

# Expected Runtime

On NVIDIA RTX A6000:

- Dataset generation: < 1 minute  
- Training: 30–60 minutes  
- Evaluation: < 1 minute  

---

# Reproducibility Notes

- All scripts use fixed random seeds where applicable  
- The environment file ensures reproducibility  
- Generated embeddings are saved and reusable  
- The code runs on GPU or CPU  

---

# Contact

Xinran Liu  (xinran.liu@vanderbilt.edu)
PhD Candidate, Computer Science  



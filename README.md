# Multimodal Spatial Representation Learning 

This repository contains code to:

1. Generate a synthetic multimodal geospatial dataset  
2. Train a shared masked graph autoencoder using self-supervised learning  
3. Evaluate learned spatial embeddings on downstream regression tasks  

For reproducibility, the full pipeline runs with a predefined default seed and requires no additional input arguments.

---

# TL;DR (Full Reproducibility Pipeline)

Run the following commands in order:

```bash
git clone https://github.com/xinranliueva/GeoEmb.git
cd GeoEmb/

# Create and activate environment (mamba strongly recommended)
mamba env create -f environment.yml
conda activate geo

# Generate dataset
python data/data_generator.py

# Train embedding model
python pretrain/pretrain_shared.py

# Evaluate embeddings
python Evaluation/eval.py
```

---

# Environment Setup

First, clone the repository:

```bash
git clone https://github.com/xinranliueva/GeoEmb.git
cd GeoEmb/
```

We provide an environment file for reproducibility.

## Create the environment

We **strongly recommend using `mamba`** for faster and more reliable environment installation. It is fully compatible with conda and significantly reduces environment solving time.

Install `mamba` in the base environment (only needed once):

```bash
conda activate base
conda install -c conda-forge mamba
```

Then create the environment:

```bash
mamba env create -f environment.yml
```

Alternatively, you can use `conda` (this may be much slower):

```bash
conda env create -f environment.yml
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


---

# Step 1: Generate Dataset

Run:

```bash
cd data
python data_generator.py 
```

By default, the data is generated at the **postal code level**. To generate data at the **county level**, use:

```bash
python data_generator.py --level county
```

This will create:

```
data/region_graph_with_features_and_targets.npz
```

This script uses a **deterministic data generator**, so it will produce the same dataset by default.

If you would like to control or modify the random seeds, please refer to the following files, which handle different components of the generation process:

- `wind.py` – wind field generation  
- `pollution.py` – pollution field generation  
- `region.py` – spatial region and graph generation  
- `target.py` – downstream target generation

Runtime: around 3 minutes.

---

# Step 2: Train Embedding Model

Run:

```bash
cd pretrain
python pretrain_shared.py --cuda 1
```
Set `--cuda` to the desired GPU index.

This will create the following files:

```
checkpoints/shared_final_emb_128.pt
checkpoints/shared_AE_128.pt
```

- `shared_final_emb_128.pt` contains the learned spatial embeddings.
- `shared_AE_128.pt` contains the trained model weights.

Typical runtime on RTX A6000:

23-28 minutes  

---

# Step 3: Evaluate Embeddings

Run:

```bash
cd Evaluation
python eval.py 
```
By default, the evaluation uses the **respiratory risk (`res`) target**.

To evaluate on the **environmental burden (`env`) target**, run:

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

**Note:** Evaluation includes grid search to select the best hyperparameters for each method, and typically takes around 11 minutes to complete.

---

# Test 
We provide unit tests to verify the correctness of the data generation and model implementation.

To run all tests, execute:

```bash
pytest -q
```

This will automatically discover and run all tests in the `test/` folder.

---

# Expected Runtime

On NVIDIA RTX A6000:

- Dataset generation: < 3 minute  
- Training: 23-28 minutes  
- Evaluation: < 11 minute  

---

# Reproducibility Notes

- All scripts use fixed random seeds where applicable  
- The data, checkpoints, and results included in this repository are generated at the **postal code level** by default.
  
---

# AI assistance and disclosure

ChatGPT was used to assist with drafting code comments, documentation, and preliminary test files. All such content was thoroughly reviewed, corrected, and validated by the author. The research ideas, methodology, implementation, experiments, and conclusions were conceived, executed, and verified by the author.

---

# Contact

Xinran Liu  (xinran.liu@vanderbilt.edu)
PhD Candidate, Computer Science  



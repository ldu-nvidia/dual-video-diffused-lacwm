# Latent Action Models

Code for training latent action models.

## Setup

> Note: this project uses the `robot_world_models` conda environment; see the setup instructions [here](/README.md).

1. Download the Cosmos tokenizer checkpoints (see: [instructions](https://github.com/NVIDIA/Cosmos/tree/main/cosmos1/models/tokenizer#download-pre-trained-checkpoints-from-hugging-face))

   Run the following script in your `COSMOS_HOME` directory:
   ```python
   from huggingface_hub import snapshot_download

   if __name__ == "__main__":
       model_names = [
           "Cosmos-0.1-Tokenizer-CI8x8",
       ]
       for model_name in model_names:
           hf_repo = "nvidia/" + model_name
           local_dir = "checkpoints/" + model_name
           print(f"downloading {model_name} to {local_dir}...")
           snapshot_download(repo_id=hf_repo, local_dir=local_dir)
   ```

2. Create a `data/datasets` folder:

   ```bash
   # create a datasets folder
   mkdir -p data/datasets

   # add symlink
   ln -s /fsx-cortex-datacache/shared/datasets/droid/011825/droid_h5 data/datasets/droid
   ```

3. Create a `data/experiments` folder:

   ```bash
   # create a folder to store experimental results; you can use any folder;
   # for example:
   mkdir -p /fsx-cortex/$USER/experiments/latent_action_models

   # add symlink
   ln -s /fsx-cortex/$USER/experiments/latent_action_models data/experiments
   ```

4. Check that the `data` directory looks something like this:

   ```
   tree -d data/
   data/
   ├── datasets
   │   └── droid -> /fsx-cortex-datacache/shared/datasets/droid/011825/droid_h5
   └── experiments -> /fsx-cortex/$USER/experiments/latent_action_models
   ```

## Using Jepa tokenizers

If you only plan to use Cosmos and DINO tokenizers you can skip this section.
However, if you plan to use Jepa tokenizers, you should install `st_wm` locally so that you can get the Jepa tokenizers and configs from huggingface. This should be later fixed to using the open-source version

```
cd projects/st_wm
pip install -e .
```

## Training

> Note: currently, experiments must be run from the project directory (`cd projects/latent_action_models`).

To debug locally, run:

```bash
HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py trainer.config.max_iter=50 trainer.config.logging.log_every=10 +experiments_0908=ravenhuang/fine-tune_0910/cosmos_st_libero_sim_scratch_v2.yaml name=t
```

To launch a multi-node job with submitit (via hydra), run:

```bash
python train.py +experiments=ravenhuang/jepa_st_lam_ae_ad_small.yaml -m &
```

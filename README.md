## Robot world models

Create a conda environment:
```bash
# option 1: create a new environment
conda env create -f environment.yaml

# option 2: clone an existing environment
# conda create --name robot_world_models --clone /fsx-cortex/sergioarnaud/miniconda/envs/robot_world_mdels
```

Install `robot_wm`, pre-commit hooks, and update submodules:
```bash
conda activate robot_world_models
pip install xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pre-commit install
git submodule update --init
```

Clone Cosmos and add the path to `COSMOS_HOME`:
```bash
cd ..  # cd out of `robot_world_models`
git clone https://github.com/NVIDIA/Cosmos.git
git switch archived-ces2025
printf "\nexport COSMOS_HOME=$PWD/Cosmos\n" >> ~/.bashrc
source ~/.bashrc
cd ../robot_world_models
```

Set huggingface environment variables so that downloaded models/data get cached in the right place:
```bash
# This is for A100 cluster, adjust the path accordingly if you're using a different cluster.
mkdir -p /fsx-cortex/$USER/huggingface/cache/datasets
mkdir -p /fsx-cortex/$USER$/huggingface/cache/hub
printf "\nexport HF_DATASETS_CACHE=/fsx-cortex/$USER/huggingface/cache/datasets\n" >> ~/.bashrc
printf "\nexport HF_HUB_CACHE=/fsx-cortex/$USER/huggingface/cache/hub\n" >> ~/.bashrc
source ~/.bashrc
```

You can run the following test to make sure things are working well.
```bash
python -m pytest -xvs robot_wm/tests/test_modules.py
python -m pytest -xvs robot_wm/tests/test_menagerie.py
```

## Codebase Structure

```
├── robot_wm
│   ├── __init__.py
│   ├── datasets
│   │   ├── base.py
│   │   ├── configs
│   │   └── droid
│   ├── inference
│   │   ├── actor
│   │   ├── config
│   │   ├── robot
│   │   ├── task
│   │   └── world_model
│   │   ├── evaluate.py
│   ├── menagerie
│   │   ├── config
│   │   ├── dino_wm.py
│   │   ├── jepa_wm.py
│   │   └── st_wm.py
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── configs
│   │   ├── modules
│   │   ├── robot_actions
│   │   └── tokenizers
│   ├── tests
│   │   ├── test_menagerie.py
│   │   ├── test_modules.py
│   │   └── ...
│   └── utils
│       ├── distributed.py
│       ├── huggingface.py
│       ├── wandb.py
│       ├── trainer.py
│       └── ...
├── examples
│   └── test_wm.ipynb
├── projects
│   ├── demo
│   ├── dino_wm
│   ├── franka_evals
│   ├── jepa-internal
│   ├── navigation_wm
│   └── st_wm
│       ├── configs
│       ├── models
│       ├── train.py
│       └── ...
└── setup.py
```

## Robot_WM


## Projects

ST WM:
- POCs: Arjun Majumdar, Sergio Arnaud
- Getting Started: See [st_wm](projects/st_wm/README.md)

DinoWM:
- POCs: Daniel Dugas, Sergio Arnaud
- Getting Started: TODO

Franka Evals:
- POCs: Abha Gejji, Daniel Dugas, Sergio Arnaud
- Getting Started: TODO

Demo:
- POCs: Phillip Thomas
- Getting Started: TODO

Navigation WM:
- POCs: Daniel Dugas
- Getting Started: TODO

Offline Evals:
- POCs: Daniel Dugas
- Getting Started: See [offline evals](offline_evals/README.md)
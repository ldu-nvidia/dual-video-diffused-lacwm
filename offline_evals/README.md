```
# Autoconvert notebook to script
cd ~/Code/robot_world_models/offline_evals && jupyter nbconvert --to python dev_offline_eval.ipynb && mv dev_offline_eval.py offline_eval.py && sed -i '/BEG_REMOVE_DEV_rqi24u/,/END_REMOVE_DEV_rqi24u/d' offline_eval.py && cd ~/Code/robot_world_models

# tmux so jobs don't get killed
tmux

# Example: Run locally
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Artem-DROID.yaml"
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Mido-DROID.yaml"

# Example: Run Calibration Evals (one per job)
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Mido-DROID.yaml" --which-eval "EvalCalibSphericalShell" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Mido-DROID.yaml" --which-eval "EvalCalibCube" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Mido-DROID.yaml" --which-eval "EvalCalibMagnitude" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Mido-DROID.yaml" --which-eval "EvalCalibOpt" --log-to-file &
# 
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Artem-DROID.yaml" --which-eval "EvalCalibSphericalShell" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Artem-DROID.yaml" --which-eval "EvalCalibCube" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Artem-DROID.yaml" --which-eval "EvalCalibMagnitude" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-Artem-DROID.yaml" --which-eval "EvalCalibOpt" --log-to-file &
# 
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-ST-1B-DROID.yaml" --which-eval "EvalCalibSphericalShell" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-ST-1B-DROID.yaml" --which-eval "EvalCalibCube" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-ST-1B-DROID.yaml" --which-eval "EvalCalibMagnitude" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/JEPAv2-ST-1B-DROID.yaml" --which-eval "EvalCalibOpt" --log-to-file &
#
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/DINOv2-ST-1B-DROID.yaml" --which-eval "EvalCalibSphericalShell" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/DINOv2-ST-1B-DROID.yaml" --which-eval "EvalCalibCube" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/DINOv2-ST-1B-DROID.yaml" --which-eval "EvalCalibMagnitude" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/DINOv2-ST-1B-DROID.yaml" --which-eval "EvalCalibOpt" --log-to-file &
#
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/CT-ST-1B-DROID.yaml" --which-eval "EvalCalibSphericalShell" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/CT-ST-1B-DROID.yaml" --which-eval "EvalCalibCube" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/CT-ST-1B-DROID.yaml" --which-eval "EvalCalibMagnitude" --log-to-file &
srun --account cortex -q cortex_high --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=12 --mem=1024G --time-min=6000 \
python offline_evals/offline_eval.py --wm-config "menagerie/config/CT-ST-1B-DROID.yaml" --which-eval "EvalCalibOpt" --log-to-file &
```
# Full process from raw MANUS and Motive recordings

`full_process.py` reads the two raw recordings for each trial and produces the requested result in your output directory. One command performs the registration, filtering and any requested model fitting and rendering. The processing code and the robot assets are included in this folder.

## Prepare the inputs

Use Python 3.12. Extract both raw data ZIPs under a common directory. These examples use `$HOME/GRASP`. For the Tripod example, the input files should be:

```text
GRASP/
  Manus_data_GRASP/14.Tripod/Tripod_R/Tripod_R_1.csv
  Motive_data_GRASP/14.Tripod/Tripod_R/Tripod_R_1.csv
```

From the directory containing this README, create a virtual environment and install the dependencies for the products you need. For all products:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For just the robot workflow you can install `python3 -m pip install -r stages/registered_to_filtered/requirements.txt -r stages/filtered_to_mujoco/requirements.txt`. For just the CSV products, `python3 -m pip install -r stages/registered_to_filtered/requirements.txt` is sufficient. To produce SMPL-X videos, obtain the neutral `SMPLX_NEUTRAL.npz` and select an appropriate PyTorch build for your machine. GPU fitting is the default. CPU fitting can be requested with `--device cpu`. [SMPL-X setup and model access](stages/filtered_to_smplx/README.md)

## Make the Tripod robot video

```bash
python3 full_process.py --raw-dir "$HOME/GRASP" --trial Tripod_R_1 \
  --target mujoco --output-dir "$HOME/GRASP/results/robot"
```

Open `$HOME/GRASP/results/robot/Tripod_R_1.mp4`. The video is 1920 × 1080 at 50 fps. The temporary registered and filtered files are removed when the trial finishes. The MuJoCo renderer needs EGL/OpenGL or OSMesa on Linux. [Robot setup](stages/filtered_to_mujoco/README.md)

## Choose the final products

| `--target` | Files saved in `--output-dir` |
| --- | --- |
| `registered` | Registered CSVs in the dataset's numbered grasp folders |
| `filtered` | Filtered CSVs and their original-time row maps in numbered grasp folders |
| `smplx` | One `<trial>.mp4` per trial. Requires `--model`. |
| `mujoco` | One `<trial>.mp4` per trial |
| `all` (default) | Separate `registered/`, `filtered/`, `smplx/` and `mujoco/` folders. Requires `--model`. |

If the two raw folders are in different locations, pass their paths separately. For example, to produce all four products for a single recording:

```bash
python3 full_process.py \
  --manus-dir "$HOME/GRASP/Manus_data_GRASP" \
  --motive-dir "$HOME/GRASP/Motive_data_GRASP" \
  --trial Tripod_R_1 --target all \
  --model "$HOME/models/SMPLX_NEUTRAL.npz" \
  --output-dir "$HOME/GRASP/results/tripod"
```

The four products appear under `$HOME/GRASP/results/tripod`. Add `--save-motion` to save the fitted model parameters in an NPZ beside each video. You can open the resulting `filtered/` folder directly in the skeleton viewer.

Repeat `--trial` to choose several recordings. Omit it to process every matched raw pair in the input folders. For the published dataset this selects 1,320 trials. The program checks the pairing and requested output paths before computing, and prints the saved paths as each trial finishes. Existing output files are protected. After an interrupted batch the finished trials remain available. Select the remaining trials or a fresh output directory when you run again.

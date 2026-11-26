# Filtered motion to SMPL-X video

Fit one filtered GRASP trial to the neutral SMPL-X body and write a 1920 × 1080 H.264 MP4. The left panel shows the full body. The right panel shows the recorded hand at a constant zoom and an orientation fixed relative to the palm in its first valid frame. The title identifies the grasp, measured hand, and trial.

## Setup

Use Python 3.12 in an isolated environment. The reference SciPy version was 1.16.1. Open a terminal in the code folder containing this README. Create and activate a virtual environment, then run the install command:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The reference fitting environment used PyTorch 2.8.0 with CUDA 12.8. Install the PyTorch build for your GPU/driver before installing the remaining dependencies. CPU fitting is available with `--device cpu`. Rendering uses a CPU triangle rasterizer and FFmpeg provided by `imageio-ffmpeg`. The renderer uses Noto Sans CJK Medium when available, then falls back to DejaVu Sans or the Pillow default.

Obtain the neutral **SMPL-X NPZ** from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) under the model terms applicable to your use. Pass the file itself to `--model`. The dataset's reference model SHA-256 is `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`. Other model releases or fitting devices can produce different motion. The package contains no licensed body-model files.

## Run a trial

Extract `Registration_data_GRASP_filtered.zip` from the [GRASP release](https://doi.org/10.5281/zenodo.24498489), or use the directory produced by the filtering workflow. The example assumes the filtered ZIP was extracted under `$HOME/GRASP` and the model file is in `$HOME/models`. From the code folder run:

```bash
python3 filtered_to_smplx.py --input-dir "$HOME/GRASP/Registration_data_GRASP_filtered" --trial Large_Diameter_R_1 \
  --model "$HOME/models/SMPLX_NEUTRAL.npz" --output-dir "$HOME/GRASP/results/human"
```

The command writes `$HOME/GRASP/results/human/Large_Diameter_R_1.mp4`. To choose a filename, use `--output videos/my_sample.mp4` in place of `--output-dir videos`.

`--input-dir` (alias `--data-root`) accepts a directory containing `Registration_data_GRASP_filtered` or the tier directory itself, including a custom output directory of the filtering workflow. For example, the trial above reads `1.Large_Diameter/Large_Diameter_R/Large_Diameter_R_1_registered.csv` and the sibling `.rowmap.csv` beneath that tier. Choose the exact dataset trial identifier, including `L` or `R` and the trial number from 1 through 20. Both hands follow the same four fitting stages.

For native parameters as well as a movie, append `--save-parameters fits/Large_Diameter_R_1.npz`. The NPZ stores body, left- and right-hand pose, global orientation and translation, face/shape parameters, measured hand, zero-based original row indices, original time in seconds, the valid-row mask, the input CSV/row map/model SHA-256 digests, and the measured-finger and participant palm-width scales. Rotations are axis-angle in radians and translations are metres. These are native SMPL-X motion parameters; the measured-finger surface profile is applied when rendering. Choose fresh output filenames. The program decodes the complete temporary movie before publishing its output.

## What is reconstructed

The fitter uses all eligible filtered samples at the original 200 Hz clock. The body seed aligns the neutral model globally, then fits the body joints in overlapping windows. The next pass fits palm orientation and measured finger chains. Two further passes refine measured wrist and digit rotations using grasp-specific fingertip and posture targets, with temporal continuity applied within each contiguous source interval. Global scale is 1.0, shape and expression coefficients are fixed at zero, and the measured finger width and thickness are both scaled to 0.7 during surface reconstruction. Digit lengths and palm length retain a scale of 1.0. The saved participant-specific palm-width factors are 0.9747079366 for S01 (trials 1–10) and 1.0120411433 for S02 (trials 11–20).

The movie samples the original timeline at 50 frames per second. It selects the nearest source frame within half a native sample. Missing times or ineligible samples leave both avatar panels blank. Keep the CSV and its row map together, especially for recordings with gaps. A rendered grasp is a kinematic reconstruction. The mesh and fingertip distances may differ from the glove measurements, and the movie does not certify physical object contact.

## Sources and licences

Dataset-specific code is covered by the accompanying MIT licence. The body-seed implementation follows the study's KITECH adaptation of [Mocap-to-SMPLX](https://github.com/LvXinTao/Mocap-to-SMPLX). The optimization and rendering constants in `functions/` capture the GRASP sample-video protocol. The vendored historical PyTorch L-BFGS implementation in `functions/body_lbfgs.py` is distributed under the PyTorch BSD-style terms in `licenses/PyTorch-LICENSE.txt`. It retains the upstream Torch line-search credits. The separate SMPL-X model and Python package retain their upstream terms. For model and method attribution see [SMPL-X](https://smpl-x.is.tue.mpg.de/) and the [official Python package](https://github.com/vchoutas/smplx).

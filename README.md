# GRASP Python workflows

Choose the folder for the data you already have and the result you want. Each folder can be copied and run on its own.

| I have | I want | Open this folder |
| --- | --- | --- |
| Raw MANUS and Motive recordings | Registered motion, filtered motion, a model video, or all four products | [01 · Full process](01_full_process/README.md) |
| Registered motion | Filtered CSVs and maps to the original sample times | [02 · Filter](02_registered_to_filtered/README.md) |
| Filtered motion | A human SMPL-X video | [03 · SMPL-X](03_filtered_to_smplx/README.md) |
| Filtered motion | A seated H1 + Shadow Hand robot video | [04 · MuJoCo](04_filtered_to_mujoco/README.md) |
| Filtered motion | Play the body and hand as a line skeleton in a desktop window | [05 · Viewer](05_skeleton_viewer/README.md) |

## Get the code and data

1. Clone this repository with `git clone https://github.com/suncheolkwon/GraspDataset_PythonCode_Workflows.git`, or download the workflow ZIP from Zenodo. Download the data ZIPs for your chosen workflow from Zenodo. For **01** get both `Manus_data_GRASP.zip` and `Motive_data_GRASP.zip`. For **02** get `Registration_data_GRASP.zip`. For **03**, **04** or **05**, get `Registration_data_GRASP_filtered.zip`. [Dataset and files on Zenodo](https://doi.org/10.5281/zenodo.24498489)
2. Extract the ZIPs. The examples assume you put the extracted folders in `$HOME/GRASP`. Keep the numbered grasp folders and the filenames inside each data tier. A filtered recording includes both a motion CSV and its sibling `.rowmap.csv`.
3. Open a terminal in the chosen code folder. For example, after cloning, run `cd GraspDataset_PythonCode_Workflows/05_skeleton_viewer` from the parent of the clone. Use Python 3.12. Run the setup below once in that folder. SMPL-X additionally needs your licensed neutral model and a suitable PyTorch build. See its setup notes before installing.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Try the viewer

For example, after the setup in folder **05**, run this command. Choose a grasp, hand and recording, then press **Play**. You can also start the viewer without arguments and use **Open dataset…**.

```bash
python3 filtered_skeleton_viewer.py --input-dir "$HOME/GRASP/Registration_data_GRASP_filtered"
```

For conversions, the selected folder's README gives a one-command example with `--output-dir`. Set it to the directory where you want the results. `Tripod_R_1` selects the first right-hand recording of Tripod. The two video workflows produce an MP4 with a full-body view on the left and the measured hand on the right.

## Licenses

The dataset is available under CC BY 4.0. GRASP workflow code is available under the MIT License. Third-party components and robot assets retain the notices in each workflow. [Dataset DOI](https://doi.org/10.5281/zenodo.24498489) · [Code license](LICENSE)

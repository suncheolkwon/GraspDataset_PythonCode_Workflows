# GRASP skeleton viewer

Open a filtered GRASP recording and play its measured 3D joint positions in a desktop window. The two panels show the complete body and the hand wearing the glove.

## Start

Use a graphical desktop and a Python 3 environment (3.10 or newer). Open a terminal in the directory containing this README. Install the dependencies once and start the viewer.

```text
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 filtered_skeleton_viewer.py
```

Choose **Open dataset…** and select the extracted `Registration_data_GRASP_filtered` folder, its parent, or the separate output directory produced by the registration-to-filtering program. Select the grasp, hand and recording, then press **Play**. The viewer lists only recordings present in the selected folder. For this dataset, takes 1–10 are participant S01 and takes 11–20 are S02.

To open the filtered tier directly, use `--input-dir`. Add `--trial` to start with a particular recording. This example assumes the filtered ZIP was extracted under `$HOME/GRASP`.

```text
python3 filtered_skeleton_viewer.py --input-dir "$HOME/GRASP/Registration_data_GRASP_filtered" --trial Small_Diameter_R_14
```

From any other working directory, pass the full path to `filtered_skeleton_viewer.py` to `python3`. The viewer reads the motion files and plays them in its window. It requires no output directory and leaves the dataset untouched. The older input option `--data-root` is also accepted.

## Inspect a movement

- Space plays or pauses. The timeline scrubs through individual retained frames. Left and Right step backward or forward by one frame.
- Choose the playback speed or enable looping with the controls below the timeline. At normal speed the viewer uses elapsed time to follow the recording. Display refresh can skip intermediate frames. Single-frame stepping exposes every stored sample.
- Drag either panel to rotate its camera, scroll to zoom and double-click to reset. The hand view follows the wrist. Its initial orientation and magnification stay fixed during playback.
- With the release's companion `.rowmap.csv`, the lower right shows the original acquisition time in seconds and the zero-based source row. In `Small_Diameter_R_14`, original row 245 is 1.225 s. The viewer checks the original time and source row against the 200 Hz clock. It requires the companion map, since the unfiltered registered tier uses the same motion filenames.

The drawing connects Motive skeleton segments and the 25 MANUS nodes of the measured hand. Motive finger segments are drawn on the opposite side. The thumb uses CMC–MCP–DIP–TIP as labeled in the release. Each point comes directly from a filtered CSV position in millimetres. The viewer displays the recorded motion without estimating a surface or a contact force.

## Dependencies and attribution

GUI and numeric reading use PyQt6 and NumPy. Each dependency is distributed under its own terms; this package contains the GRASP viewer source. Based on the dataset conventions and the student's `Visualize_Registration.py`. See `LICENSE` for the source notice and dataset license.

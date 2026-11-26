# Replay a filtered grasp on the H1 with Shadow hands

This program fits a seated humanoid to the selected filtered recording and writes an H.264 video with a whole-body view and a close-up of the measured hand. Each panel is 960 × 1080 pixels. The hand camera follows the palm's translation and keeps the viewing angle from the first frame.

## Run one trial

Extract the published `Registration_data_GRASP_filtered.zip` or use the output of the filtering program. Keep the numbered grasp folders and each CSV's companion `.rowmap.csv` together. The commands below work from the directory containing this README, using a Python 3.12 environment. The example assumes the input ZIP was extracted under `$HOME/GRASP`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 filtered_to_mujoco.py \
  --input-dir "$HOME/GRASP/Registration_data_GRASP_filtered" \
  --output-dir "$HOME/GRASP/videos" \
  --trial Tripod_R_1
```

The program creates `$HOME/GRASP/videos/Tripod_R_1.mp4`. To run the same script from another working directory, give `python3` the full path to `filtered_to_mujoco.py`. It locates its helpers and robot meshes next to the script.

Change `--trial` to choose a recording. For example, `Large_Diameter_L_1` fits and displays the physical left Shadow Hand. To choose an exact filename, replace `--output-dir` with `--output /path/to/video.mp4`. Select a fresh output filename for each run. The `--input-dir` option also accepts the filtered tier's parent directory or a renamed filtered tier. Its older spelling `--data-root` works as well.

The robot model, meshes and scene definitions are supplied in this package. Fitting runs on the CPU. On Linux the script defaults to offscreen EGL rendering and automatically locates the system EGL driver. Systems configured for OSMesa can set `MUJOCO_GL=osmesa`. Video encoding automatically uses the FFmpeg executable from the imageio-ffmpeg wheel. To use a separately installed FFmpeg encoder, set `IMAGEIO_FFMPEG_EXE` to its executable path.

## Output and validation

The program solves all native 200 Hz samples. The MP4 shows every fourth sample at 50 fps. The acquisition time is recovered from the row map. Optional `--trajectory Tripod_R_1.npz` saves the floating-base and joint coordinates, original timestamps and registered row numbers, CSV SHA-256, video-to-source indices, joint names and their qpos addresses. The 76 qpos coordinates begin with pelvis XYZ (metres) and WXYZ quaternion, followed by 69 hinge coordinates (radians). MuJoCo's MJCF model defines their order.

For each original frame, forward kinematics independently recalculates all ten fingertip distances, the palm normal and the deepest active robot self-collision. The program prints a summary. The optional NPZ includes the absolute fingertip-distance errors in mm, the palm-normal angle in degrees and the deepest overlap in mm. The ten pairs follow the order of all combinations of thumb, index, middle, ring and little finger.

Release checks covered four right-hand trials (`Small_Diameter_R_1`, `Tripod_R_1`, `Extension_Type_R_1` and `Palmar_R_1`) and one left-hand trial (`Large_Diameter_L_1`). Each movie was decoded in full. On `Tripod_R_1` the program reproduced the reference robot coordinates exactly across all 1,328 source frames. On the left-hand trial, the fitted left fingers moved, the right fingers remained at neutral and the mean absolute error across all ten fingertip distances and frames was 1.70 mm. These checks cover the five listed trials.

## Motion model

- The floating pelvis follows the measured hip midpoints and trunk lean. A constant vertical translation places the lowest sole on the virtual floor. The body fit tracks the participant's upper-arm, forearm, thigh, shin and foot directions using the robot's native link lengths and joint limits.
- The measured palm normal and wrist-to-knuckle direction set the target for the measured arm. A hand-local fit tracks all ten unscaled fingertip distances, relative tip vectors and finger-bone directions. Analytic MuJoCo Jacobians and bounded least squares determine the joint coordinates. The four non-thumb distal hinge pairs share their tendon flexion equally. The other hand's fingers stay at neutral.
- The original right-hand Tripod demonstration retains its tested shoulder initialization. Other trials screen arm branches over the motion and check finger continuity against changes in the measured fingers. Extension Type, Distal Type, Palmar and Ventral also evaluate bounded torso-yaw adjustments. Palmar evaluates a grasp-centred branch from the middle of the recording. Where needed, the program tests small movements of the ungloved arm and the two hip shells to reduce crossings.
- This is kinematic motion replay. The visible chair is fitted to the seated pose. Object contacts, forces, balance under gravity and hardware motor execution are outside this model. Wrist reach and individual finger angles can differ from the human measurements even when the fingertip distances match. Use the per-frame values with the original recordings when assessing a trial.

## Models and licenses

The unmodified model files and included meshes come from HumanoidBench revision `cb1189039151c8aadaaa987b442da54383c87fab`. The upstream snapshot contains the H1 body with two articulated Shadow hands. Original notices are in `robot_assets/LICENSE`, `robot_assets/h1/LICENSE` and `robot_assets/shadow_hand_menagerie/LICENSE`. The dataset processing code uses the MIT license supplied in `LICENSE`. [Pinned robot source](https://github.com/carlosferrazza/humanoid-bench/tree/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/assets)

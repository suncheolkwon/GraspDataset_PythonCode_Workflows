# Filter registered GRASP recordings

Run `registered_to_filtered.py` once to convert registered hand and body kinematics into the **filtered** tier of the GRASP Taxonomy Hand Motion Dataset. This implements the released event-preserving method (development name `filtered_v3`). The only motion input is the registered CSV. The side-specific skeleton and sample ranges supplied with the code cover the 1,320 recordings.

## Setup

Download `Registration_data_GRASP.zip` from the dataset record and extract it into `$HOME/GRASP`. Your input directory is `$HOME/GRASP/Registration_data_GRASP`. For a fresh installation, open a terminal in the `02_registered_to_filtered` code folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Use Python 3.12 for the tested numerical environment.

## Run the filter

The following command filters all registered recordings present in the chosen input directory and creates the destination if needed:

```bash
python3 registered_to_filtered.py --input-dir "$HOME/GRASP/Registration_data_GRASP" --output-dir "$HOME/GRASP/rebuilt_filtered"
```

To filter one recording, add its exact name. You can repeat `--trial` for multiple recordings:

```bash
python3 registered_to_filtered.py --input-dir "$HOME/GRASP/Registration_data_GRASP" --output-dir "$HOME/GRASP/rebuilt_filtered" --trial Small_Diameter_R_14
```

`--input-dir` can also point to a single grasp folder, a folder of selected recording files, or a registered folder produced by the full-process program. The program searches subfolders for the release's `*_registered.csv` filenames. Exact trial names must be unique. Each supplied trial retains the release's column names and registered row count. Add `--require-complete` to require all 1,320 trials and `--workers 4` to process four in parallel.

The program saves files in numbered grasp and hand folders under `--output-dir`. For the example above, the two outputs are:

```text
rebuilt_filtered/2.Small_Diameter/Small_Diameter_R/Small_Diameter_R_14_registered.csv
rebuilt_filtered/2.Small_Diameter/Small_Diameter_R/Small_Diameter_R_14_registered.rowmap.csv
```

The first file contains the 534 motion columns. The second maps each filtered frame to the original recording. Choose a destination separate from the inputs. On a repeated run, the program checks existing pairs, fills missing files and reports mismatches with a nonzero exit status.

## Compare with the distributed files

To compare with the published filtered data, also extract `Registration_data_GRASP_filtered.zip` under `$HOME/GRASP`. Run:

```bash
python3 registered_to_filtered.py --input-dir "$HOME/GRASP/Registration_data_GRASP" --verify-against "$HOME/GRASP/Registration_data_GRASP_filtered" --require-complete --workers 4
```

This mode recomputes the motion from registered samples, then compares every output cell and every source-row map. Frame labels, time labels and correspondence maps must match exactly. Tolerances are 10⁻⁸ mm per position component and 10⁻⁹ per signed quaternion component. The final JSON reports the number of trials checked and the observed maxima.

With Python 3.12.11 and the pinned libraries, all 1,320 released registered inputs reproduced 1,874,713 filtered rows. Every position and signed quaternion component agreed exactly after reading the published CSVs at round-trip precision. Every source-row map also agreed exactly. In eight representative recordings, including all four exceptional start rows, all 16 written CSV and map files matched the published SHA-256 checksums.

## What the filter does

1. Use the retained samples for this trial. For each of 24 MANUS hand links and 10 optical arm/leg links, measure the mean registered length. Reconstruct the skeleton with those lengths while retaining the measured bone directions. This per-trial reconstruction is the baseline for motion protection.
2. Compare consecutive samples in the moving wrist frame. A joint displacement above 3 mm, a change in a fingertip-pair distance above 3 mm, or a parent-relative rotation above 3° protects both samples and two neighbors on either side. Fast movement therefore keeps its original timing and its length-corrected baseline trajectory.
3. Smooth distal finger directions only where the complete centered window is quiet. The numerical kernel is `[1, 2, 3, 2, 1] / 9`. Each quiet label itself requires 21 consecutive frame-to-frame steps below 0.5 mm of wrist-local joint movement and 1° of relative rotation. The largest correction to any hand point is capped at 0.25 mm relative to the baseline. Each quiet run also limits the peak and RMS of the five fingertips' second differences. Hand rotations follow the change in their outgoing bones.
4. Export the filtered coordinates on the 200 Hz grid. The sidecar preserves the original registered row and timestamp for every output row. Its `source_time_s` is the time to use when comparing with raw or registered recordings. The main motion file's `Time` starts at zero.

The distributed time ranges retain all samples from the first included row to the end. The zero-based first row is 5 in 1,316 trials. The four exceptions are `Small_Diameter_L_1` (37), `Small_Diameter_L_4` (9), `Small_Diameter_L_5` (15), and `Index_Finger_Extension_R_1` (65). For example, registered row 245 at 1.225 s in `Small_Diameter_R_14` appears at filtered row 240, whose displayed time is 1.200 s.

The numerical core is in `grasp_filter/filter.py` and `grasp_filter/kinematics.py`. The two small CSVs in that directory contain node parents and trial correspondence metadata. These files contain no precomputed motion coordinates. This is an offline method designed for this release's recordings. Low-amplitude measured motion can meet its numerical definition of quiet.

## License

See `LICENSE` for the code license. [GRASP Taxonomy Hand Motion Dataset](https://doi.org/10.5281/zenodo.24498489)

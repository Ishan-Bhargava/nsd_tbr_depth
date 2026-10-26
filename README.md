# Tyre tread-depth estimator (TBR)

Upload a short video of a truck-tyre tread and get the tread depth in millimetres.

- **Model:** DINOv3 ViT-B/16 (via `timm`) at 336 px, 16 frames per video, spatial + temporal attention pooling,
  regression head. Final answer = average of a **5-fold ensemble**; the spread between the 5 models is shown as a
  confidence signal.
- **Trained range:** 7-21 mm. A prediction that lands outside that range is flagged as unreliable.

## Quick start

```bash
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch. On a machine with an NVIDIA GPU, install the CUDA build of
`torch` first (see pytorch.org) and then run the line above.

Put the five checkpoints in a `checkpoints_best/` folder next to the scripts (they are **not** in this repo -
they are 343 MB each):

```
checkpoints_best/
  dinov3_vitb16_depth_fold1_best.pt
  ...
  dinov3_vitb16_depth_fold5_best.pt
```

(or use another folder: the `TBR_CHECKPOINT_DIR` environment variable, `--checkpoints` on the command line, or the
folder browser in the app).

### Web app (Streamlit)
On Windows you can just double-click `run_app.bat` (it checks the packages first). Or:
```bash
streamlit run streamlit_app.py
# others on your network:  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

**Choosing the model folder.** Open the app at `http://localhost:8501` on the computer that runs it and use
*Change folder...* in the sidebar to browse to the folder with the five `.pt` files (type a path, go up, open
sub-folders; the page shows how many of the 5 files it found). The browser is shown **only** to someone on that same
computer: anyone reaching the app through a network address sees just the default folder's status, never any paths.
Set `TBR_ALLOW_BROWSE=0` to switch it off completely on shared deployments.

### Command line
```bash
python infer_tbr_dino_336.py --input tyre.mp4                  # one video
python infer_tbr_dino_336.py --input videos_folder --output preds.csv
python infer_tbr_dino_336.py --input labels.csv                # CSV with filename/video_path (+ depth_mm for an accuracy report)
```
Outputs the ensemble depth, each fold's prediction, the spread between folds and flags
(`LOW_CONFIDENCE`, `OUT_OF_RANGE`, `NO_FRAMES`).

## Hosting notes

Settings are read from environment variables (on a hosting service: its "secrets" / environment settings):

| Variable | Purpose |
|---|---|
| `TBR_CHECKPOINT_DIR` | folder with the five `.pt` files (default: `checkpoints_best` next to the scripts) |
| `TBR_WEIGHTS_BASE_URL` | download the weights at start-up if they are missing: `<base>/dinov3_vitb16_depth_fold1_best.pt` ... must be reachable. Downloads go only into the default folder. |
| `TBR_WEIGHTS_TOKEN` | optional bearer token for private storage (never forwarded to a different host on redirects) |
| `APP_PASSWORD` | if set, the page shows nothing until this password is entered - use it for any URL others can reach |
| `TBR_ALLOW_BROWSE` | `0` switches the folder browser off completely |

Put a `SHA256SUMS` file next to the hosted weights (`sha256sum *.pt > SHA256SUMS`, or `Get-FileHash` on Windows) and
every download is verified.

**Resources.** With all 5 models loaded the app uses about **2.5 GB of RAM** on CPU (measured), plus whatever the
uploaded video needs. Free hosting tiers with a memory cap near 2-3 GB (Streamlit Community Cloud documents a
maximum of about 2.7 GB and 2 CPU cores) are therefore very tight, and will also be slow. Use a machine with at least
8 GB of RAM, or a GPU host.

## Speed
About 1-2 minutes per video on a 12-thread CPU with all 5 models; expected a few seconds on a GPU (T4 class).

## Limitations
- Estimates only; not a certified measurement. Single videos can be off by more than the typical error (about 1 mm
  on held-out data in testing).
- Accuracy is weakest near the ends of the 7-21 mm range. Tyres outside the range are not handled: the model tends to
  pull them toward the middle (a very worn tyre can read much deeper than it is), so the out-of-range flag will not
  always catch them.
- Training labels are rounded to whole millimetres and a few are wrong, which limits how accurate the model can be
  measured to be.
- Videos are processed on the machine that runs the app (on a hosting service, that service's servers).

## Licence note
The backbone is initialised from Meta's DINOv3 weights, which carry their own licence terms - check them before
redistributing trained weights.

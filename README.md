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

Put the five checkpoints in a `checkpoints_best/` folder next to the scripts (they are **not** in this repo -
they are 343 MB each):

```
checkpoints_best/
  dinov3_vitb16_depth_fold1_best.pt
  ...
  dinov3_vitb16_depth_fold5_best.pt
```

(or point to another folder with the `TBR_CHECKPOINT_DIR` environment variable / `--checkpoints`; on a server the
app can also download them for you - see *Deploy on Replit*).

### Web app
On Windows you can just double-click `run_app.bat` (it checks the packages first). Or:
```bash
streamlit run streamlit_app.py
# others on your network:  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

**Choosing the model folder.** Open the app at `http://localhost:8501` on the computer that runs it and use
*Change folder...* in the sidebar to browse to the folder with the five `.pt` files (type a path, go up, open
sub-folders; the page shows how many of the 5 files it found). The browser is shown **only** to someone on that same
computer: anyone reaching the app through a network address sees just the default folder's status, never any paths.
Set `TBR_ALLOW_BROWSE=0` to switch it off completely on shared deployments (the Replit config does).

### Command line
```bash
python infer_tbr_dino_336.py --input tyre.mp4                  # one video
python infer_tbr_dino_336.py --input videos_folder --output preds.csv
python infer_tbr_dino_336.py --input labels.csv                # CSV with filename/video_path (+ depth_mm for an accuracy report)
```
Outputs the ensemble depth, each fold's prediction, the spread between folds and flags
(`LOW_CONFIDENCE`, `OUT_OF_RANGE`, `NO_FRAMES`).

## Deploy on Replit (CPU, good for demos)

1. In Replit: *Create App -> Import from GitHub* and pick this repo. `.replit` already has the run command and port.
2. Get the 5 weight files onto the machine - either way works:
   - **Download at start-up (recommended).** Host the five `.pt` files at any HTTPS location (private storage, a release
     asset, ...) so that `<base>/dinov3_vitb16_depth_fold1_best.pt` etc. resolve, and add Replit **Secrets**:
     `TBR_WEIGHTS_BASE_URL` = that base URL, and `TBR_WEIGHTS_TOKEN` if the storage needs a bearer token.
     Optionally put a `SHA256SUMS` file next to them (`sha256sum *.pt > SHA256SUMS`, or `Get-FileHash` on Windows)
     and every download is verified.
   - **Or upload by hand** into a `checkpoints_best/` folder using Replit's file panel.
3. Add a Secret `APP_PASSWORD` - a deployed Replit app has a public URL, and with this set nothing is shown until the
   password is entered.
4. *Publish -> Reserved VM* (the models stay loaded; Autoscale would reload them after every idle period). Pick a machine
   with plenty of vCPUs (8+) and at least 16 GB of memory - there is no GPU, so more vCPUs means faster answers
   (roughly 1-2 minutes per video on a 12-core CPU).

The first prediction after a fresh start loads (and, if needed, downloads) the models, so it is slower.

## Speed
About 1-2 minutes per video on CPU with all 5 models; expected a few seconds on a GPU (T4 class).

## Limitations
- Estimates only; not a certified measurement. Single videos can be off by more than the typical error (about 1 mm
  on held-out data in testing).
- Accuracy is weakest near the ends of the 7-21 mm range. Tyres outside the range are not handled: the model tends to
  pull them toward the middle (a very worn tyre can read much deeper than it is), so the out-of-range flag will not
  always catch them.
- Training labels are rounded to whole millimetres and a few are wrong, which limits how accurate the model can be
  measured to be.
- Videos are processed on the machine running the app; nothing is sent elsewhere by this code.

## Licence note
The backbone is initialised from Meta's DINOv3 weights, which carry their own licence terms - check them before
redistributing trained weights.

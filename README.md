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

(or point to another folder with the `TBR_CHECKPOINT_DIR` environment variable / `--checkpoints`).

### Web app
```bash
streamlit run streamlit_app.py
# others on your network:  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### Command line
```bash
python infer_tbr_dino_336.py --input tyre.mp4                  # one video
python infer_tbr_dino_336.py --input videos_folder --output preds.csv
python infer_tbr_dino_336.py --input labels.csv                # CSV with filename/video_path (+ depth_mm for an accuracy report)
```
Outputs the ensemble depth, each fold's prediction, the spread between folds and flags
(`LOW_CONFIDENCE`, `OUT_OF_RANGE`, `NO_FRAMES`).

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

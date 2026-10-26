"""
Tyre tread-depth tester -- upload a video, get the depth in mm.

Run:      streamlit run streamlit_app.py
Network:  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501   (others open http://<your-ip>:8501)

Uses the 5-fold DINOv3 ViT-B / 336 px ensemble through infer_tbr_dino_336.py (must sit in the same folder).
Checkpoint folder: set the TBR_CHECKPOINT_DIR environment variable, or edit DEFAULT_CHECKPOINT_DIR below.
"""

import hmac
import io
import os
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

import fetch_weights
import infer_tbr_dino_336 as core

DEFAULT_CHECKPOINT_DIR = os.environ.get("TBR_CHECKPOINT_DIR", core.DEFAULT_CHECKPOINT_DIR)
UPLOAD_TYPES = [e.lstrip(".") for e in core.VIDEO_EXTENSIONS]
LO, HI = core.TRAINED_RANGE_MM

st.set_page_config(page_title="Tyre tread depth", page_icon="🛞", layout="centered")


def password_gate():
    """If the APP_PASSWORD environment variable (a Replit Secret) is set, nothing is shown until it is entered.
    Recommended for any deployment with a public URL."""
    expected = os.environ.get("APP_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return
    st.title("🛞 Tyre tread depth")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


password_gate()


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the 5 models (first start only; downloads the weights if they are missing)...")
def get_models(checkpoint_dir: str, device_str: str):
    try:
        fetch_weights.ensure_weights(checkpoint_dir)      # no-op when the 5 files are already there
        return core.load_models(checkpoint_dir, None, torch.device(device_str))
    except SystemExit as e:                       # load_models exits with a message on problems
        raise RuntimeError(str(e))


@st.cache_resource
def get_lock():
    return threading.Lock()                       # one prediction at a time (CPU/GPU is shared by all visitors)


def predict(video_path: str, models: dict, fold_ids, device, tta: bool):
    prep = core.prepare_video(video_path)
    if prep is None:
        return None
    frames_rgb, x, fq = prep
    use_amp = core.USE_AMP_ON_CUDA and device.type == "cuda"
    x, fq = x.to(device), fq.to(device)
    preds, spatial_all, temporal_all = {}, [], []
    with torch.no_grad():
        for f in fold_ids:
            out, sp, tp = models[f](x, fq, use_amp=use_amp)
            p = float(out.item())
            if tta:
                out2, _, _ = models[f](torch.flip(x, dims=[4]), fq, use_amp=use_amp)
                p = 0.5 * (p + float(out2.item()))
            preds[f] = p
            spatial_all.append(sp.squeeze(0).squeeze(1).cpu().numpy())
            temporal_all.append(tp.squeeze(0).squeeze(-1).cpu().numpy())
    vals = np.array(list(preds.values()))
    return {
        "depth": float(vals.mean()), "std": float(vals.std()), "preds": preds,
        "frames": frames_rgb, "spatial": np.mean(spatial_all, axis=0), "temporal": np.mean(temporal_all, axis=0),
    }


def attention_png(res, name):
    path = os.path.join(tempfile.gettempdir(), f"att_{abs(hash(name + str(time.time())))}.png")
    core.save_attention_image(res["frames"], res["spatial"], res["temporal"],
                              f"{name} | predicted {res['depth']:.1f} mm", path)
    with open(path, "rb") as fh:
        data = fh.read()
    os.remove(path)
    return data


# ---------------------------------------------------------------- sidebar
device_str = "cuda" if torch.cuda.is_available() else "cpu"
with st.sidebar:
    st.header("Settings")
    st.caption(f"Running on **{'GPU' if device_str == 'cuda' else 'CPU'}**")
    ckpt_dir = DEFAULT_CHECKPOINT_DIR                     # set with the TBR_CHECKPOINT_DIR environment variable
    st.caption("All 5 models are always used; the answer is their average.")
    tta = st.checkbox("Flip check (average with a mirrored copy)", value=False, help="Twice as slow.")
    show_attention = st.checkbox("Show where the model looked", value=True)

# ---------------------------------------------------------------- main page
st.title("🛞 Tyre tread depth")
st.write("Upload a short video of the tyre tread. The model returns the tread depth in millimetres.")
st.caption("Tip: film the tread straight-on in good light, moving slowly along it for about 5–15 seconds.")

uploaded = st.file_uploader("Tyre video", type=UPLOAD_TYPES, accept_multiple_files=False)

if uploaded is not None:
    st.video(uploaded)
    run = st.button("Measure tread depth", type="primary", width="stretch")

    if run:
        try:
            models = get_models(ckpt_dir, device_str)
        except ModuleNotFoundError as e:
            # Most common cause: `pip` installed into a different Python than the one running Streamlit.
            st.error(f"A required Python package is missing: **{e.name}**")
            st.write("Install it into the SAME Python that is running this app, by running this in a terminal, "
                     "then restart the app:")
            st.code(f'"{sys.executable}" -m pip install {e.name}', language="bash")
            st.stop()
        except Exception as e:                         # noqa: BLE001
            st.error(f"Could not load the models: {e}")
            st.stop()

        fold_ids = sorted(models)                      # always every available fold
        if len(fold_ids) != 5:
            st.warning(f"Expected 5 models but found {len(fold_ids)} in {ckpt_dir}. Check the checkpoint folder.")

        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        t0 = time.time()
        try:
            lock = get_lock()
            if lock.locked():
                st.info("Another video is being processed -- yours is queued and will start automatically.")
            with lock, st.spinner(f"Analysing the video with {len(fold_ids)} model(s)..."):
                res = predict(tmp_path, models, fold_ids, torch.device(device_str), tta)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if res is None:
            st.error("Could not read any frames from this video. Try another file or format (mp4 works best).")
            st.stop()

        elapsed = time.time() - t0
        depth, spread = res["depth"], res["std"]

        # ---- the answer
        st.divider()
        st.metric("Tread depth", f"{depth:.1f} mm")

        if not (LO <= depth < HI):
            st.error("This result is outside the range the model was trained on, so treat it as unreliable.")
        elif spread > core.SPREAD_WARN_MM:
            st.warning(f"Low confidence: the models disagree by about ±{spread:.1f} mm. Try filming again in better light, "
                       f"with the tread in focus.")
        elif spread > 0.5:
            st.info(f"Medium confidence: the models differ by about ±{spread:.1f} mm.")
        else:
            st.success(f"High confidence: the models agree (±{spread:.1f} mm).")
        st.caption("In testing, results were typically within about 1 mm of the measured depth, but single videos can be off by more.")

        # ---- details
        with st.expander("Details"):
            st.write(f"Processed in {elapsed:.1f} s on {'GPU' if device_str == 'cuda' else 'CPU'}.")
            table = pd.DataFrame({"fold": [f"fold {f}" for f in res["preds"]],
                                  "prediction (mm)": [round(v, 2) for v in res["preds"].values()]})
            st.dataframe(table, hide_index=True, width="stretch")
            st.bar_chart(table.set_index("fold"))
            csv = pd.DataFrame([{"video": uploaded.name, "depth_mm": round(depth, 2), "fold_std_mm": round(spread, 2),
                                 **{f"fold{f}_mm": round(v, 2) for f, v in res["preds"].items()}}]).to_csv(index=False)
            st.download_button("Download result (CSV)", csv, file_name="tread_depth_result.csv", mime="text/csv")

        if show_attention:
            with st.expander("Where the model looked"):
                st.caption("Red/yellow = areas the model relied on most; blue = ignored. It should be on the tread blocks and grooves, "
                           "not on the sidewall, wheel or background.")
                st.image(attention_png(res, uploaded.name), width="stretch")
else:
    st.info("Upload a video to begin.")

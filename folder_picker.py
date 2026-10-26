"""
Sidebar folder browser for choosing the folder that holds the model checkpoints.

Streamlit has no native folder picker (the file uploader sends files, not paths), so this browses the folders of the
machine that RUNS the app. That is only safe for the person sitting at that machine, so the browser is shown only
when the page is opened from the same computer (http://localhost:8501). Anyone reaching the app through a network
address / proxy just gets the default folder and cannot see or change server paths.

  TBR_ALLOW_BROWSE=0   switch the browser off completely (set this on any shared or public deployment)
"""

import os
import string
import tempfile

import streamlit as st

import fetch_weights as fw

_LOCAL_IPS = {None, "127.0.0.1", "::1", "::ffff:127.0.0.1"}


def client_is_local() -> bool:
    """True only when the browser is on the same computer as the Streamlit server."""
    if os.environ.get("TBR_ALLOW_BROWSE") == "0":
        return False
    try:
        headers = st.context.headers
        if headers.get("X-Forwarded-For") or headers.get("Forwarded") or headers.get("X-Real-Ip"):
            return False                              # behind a proxy -> treat as remote
        return st.context.ip_address in _LOCAL_IPS    # None when opened via localhost
    except Exception:                                 # noqa: BLE001  (no request context -> be safe)
        return False


def list_drives():
    if os.name == "nt":
        return [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    return ["/"]


def list_subdirs(path):
    """Sorted sub-folder names (hidden/system ones skipped), or None if the folder cannot be read."""
    try:
        names = []
        for entry in os.scandir(path):
            try:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith((".", "$")):
                    names.append(entry.name)
            except OSError:
                continue
        return sorted(names, key=str.lower)
    except OSError:
        return None


def weights_present(path):
    """Which of the 5 fold checkpoints exist in this folder."""
    return [f for f in fw.FOLDS if os.path.isfile(os.path.join(path, fw.weight_filename(f)))]


# Where model files uploaded through the page are kept on the server (shared by every visitor, so only the first
# person has to upload them; it lasts until the server's temp folder is cleared or the app is restarted).
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "tbr_uploaded_models")


def folder_stamp(path):
    """Changes whenever one of the 5 checkpoint files is replaced -> lets the model cache notice new files."""
    out = []
    for f in fw.FOLDS:
        p = os.path.join(path, fw.weight_filename(f))
        try:
            st_ = os.stat(p)
            out.append((f, st_.st_size, st_.st_mtime_ns))
        except OSError:
            out.append((f, 0, 0))
    return tuple(out)


def save_uploaded_models(files, dest=None):
    """Save the 5 uploaded checkpoint files into dest (default UPLOAD_DIR). Returns (ok, message).
    All 5 must be selected together."""
    dest = dest or UPLOAD_DIR
    wanted = {fw.weight_filename(f) for f in fw.FOLDS}
    by_name, ignored = {}, []
    for uf in files:
        name = os.path.basename(str(uf.name))
        (by_name.__setitem__(name, uf) if name in wanted else ignored.append(name))
    if not by_name:
        return False, ("None of the selected files has an expected name "
                       f"({fw.weight_filename(1)} ... {fw.weight_filename(5)}).")
    missing = sorted(wanted - set(by_name))
    if missing:
        return False, f"Received {len(by_name)} of 5 files so far - select all five together (still missing: {len(missing)})."
    os.makedirs(dest, exist_ok=True)
    for name, uf in by_name.items():
        part = os.path.join(dest, name + ".part")
        with open(part, "wb") as fh:
            fh.write(uf.getbuffer())
        os.replace(part, os.path.join(dest, name))        # atomic: a half-written file is never used
    note = f" (ignored {len(ignored)} other file(s))" if ignored else ""
    return True, "Model files uploaded" + note


def _model_uploader_widget(key):
    nonce = st.session_state.get(f"{key}_up_nonce", 0)
    return st.file_uploader("Select the 5 checkpoint files (.pt) from your computer", type=["pt"],
                            accept_multiple_files=True, key=f"{key}_up_{nonce}")


def _remote_upload_ui(key, sel_k):
    """For visitors who are NOT at the server: let them provide the checkpoints through the browser's file dialog.
    Only offered when APP_PASSWORD is set (the password gate has already been passed), so strangers cannot replace
    the models on a public URL."""
    if not os.environ.get("APP_PASSWORD"):
        st.caption("To upload model files from your computer here, set an `APP_PASSWORD` in the app's settings first "
                   "(so only people with the password can replace the models).")
        return st.session_state[sel_k]
    flash = st.session_state.pop(f"{key}_flash", None)
    if flash:
        st.success(flash)
    with st.expander("Upload model files from your computer…", expanded=(len(weights_present(st.session_state[sel_k])) == 0)):
        st.caption("Browse to the folder on your computer, select all 5 `.pt` files in one go. Each is ~343 MB, so this "
                   "takes a few minutes, and the server needs enough free memory for them.")
        files = _model_uploader_widget(key)
        if files:
            ok, msg = save_uploaded_models(files)
            if ok:
                st.session_state[sel_k] = UPLOAD_DIR
                st.session_state[f"{key}_up_nonce"] = st.session_state.get(f"{key}_up_nonce", 0) + 1   # clears the widget, freeing RAM
                st.session_state[f"{key}_flash"] = "Model files uploaded - they are ready to use."
                st.rerun()
            else:
                st.info(msg)
    return st.session_state[sel_k]


def _status(path):
    n = len(weights_present(path))
    return n, (f"{n} of 5 model files found" if n else "no model files here")


def _nearest_existing_dir(path):
    path = os.path.abspath(path or ".")
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path if os.path.isdir(path) else os.path.expanduser("~")


def folder_picker(key: str, default_dir: str) -> str:
    """Draws the 'Model folder' block in the current container and returns the selected folder."""
    sel_k, cur_k, txt_k, sub_k, drv_k = (f"{key}_{s}" for s in ("selected", "browse", "path_text", "subfolder", "drive"))
    if sel_k not in st.session_state:
        st.session_state[sel_k] = default_dir
    selected = st.session_state[sel_k]
    # models uploaded earlier through the page (by anyone with the password) are used when the default folder is empty
    if (selected == default_dir and len(weights_present(selected)) < 5 and len(weights_present(UPLOAD_DIR)) == 5):
        st.session_state[sel_k] = selected = UPLOAD_DIR

    n_sel, text = _status(selected)
    local = client_is_local()
    st.markdown("**Model folder**")
    if local:
        st.caption(f"`{selected}`")
    will_download = (n_sel < 5 and bool(os.environ.get("TBR_WEIGHTS_BASE_URL"))
                     and os.path.abspath(selected) == os.path.abspath(default_dir))
    if will_download:
        st.info("The model files are downloaded the first time you press Measure "
                "(about 1.7 GB, so that first run takes a few minutes).")
    elif n_sel == 0 and not local:
        st.error("The model files are not installed on this server.")
    else:
        (st.success if n_sel == 5 else st.warning if n_sel else st.error)(text)

    if not local:
        return _remote_upload_ui(key, sel_k)

    # ---------------- in-app browser (local visitors only)
    if cur_k not in st.session_state:
        st.session_state[cur_k] = _nearest_existing_dir(selected)
    st.session_state.setdefault(txt_k, st.session_state[cur_k])

    def go(path):                                     # used inside callbacks: update the view and the text box
        path = _nearest_existing_dir(path)
        st.session_state[cur_k] = path
        st.session_state[txt_k] = path

    def on_text():
        go(st.session_state[txt_k])

    def on_up():
        go(os.path.dirname(st.session_state[cur_k].rstrip("\\/")) or st.session_state[cur_k])

    def on_sub():
        pick = st.session_state[sub_k]
        if pick and pick != "—":
            go(os.path.join(st.session_state[cur_k], pick))
        st.session_state[sub_k] = "—"

    def on_drive():
        go(st.session_state[drv_k])

    def on_use():
        st.session_state[sel_k] = st.session_state[cur_k]

    with st.expander("Change folder…", expanded=(n_sel == 0)):
        st.text_input("Path (type or paste, then Enter)", key=txt_k, on_change=on_text)
        c1, c2, c3 = st.columns(3)
        c1.button("⬆ Up", on_click=on_up, key=f"{key}_up", width="stretch")
        c2.button("Home", on_click=lambda: go(os.path.expanduser("~")), key=f"{key}_home", width="stretch")
        c3.button("Reset", on_click=lambda: go(default_dir), key=f"{key}_default", width="stretch",
                  help="Back to the default model folder")
        if os.name == "nt":
            drives = list_drives()
            cur = st.session_state[cur_k]
            idx = next((i for i, d in enumerate(drives) if cur.upper().startswith(d.upper())), 0)
            st.selectbox("Drive", drives, index=idx, key=drv_k, on_change=on_drive)

        cur = st.session_state[cur_k]
        subs = list_subdirs(cur)
        if subs is None:
            st.warning("This folder can't be read (permission denied).")
            subs = []
        st.selectbox(f"Open a sub-folder ({len(subs)})", ["—"] + subs, key=sub_k, on_change=on_sub)

        n_cur, text_cur = _status(cur)
        st.caption(f"Browsing: `{cur}`")
        (st.success if n_cur == 5 else st.info)(f"This folder: {text_cur}")
        st.button("Use this folder", type="primary" if n_cur else "secondary", on_click=on_use,
                  key=f"{key}_use", width="stretch", disabled=(n_cur == 0))
    return st.session_state[sel_k]

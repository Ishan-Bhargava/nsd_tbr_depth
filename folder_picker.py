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

    n_sel, text = _status(selected)
    local = client_is_local()
    st.markdown("**Model folder**")
    if local:
        st.caption(f"`{selected}`")
    (st.success if n_sel == 5 else st.warning if n_sel else st.error)(text)

    if not local:
        return selected

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

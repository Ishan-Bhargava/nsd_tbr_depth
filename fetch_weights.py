"""
Download the model weights on first start (they are too big for git: 5 x 343 MB).

Set these environment variables (on Replit: the "Secrets" tool):
  TBR_WEIGHTS_BASE_URL   folder-style URL; the files  dinov3_vitb16_depth_fold{1..5}_best.pt  must be reachable as
                         <BASE_URL>/dinov3_vitb16_depth_fold1_best.pt  etc.
                         e.g. https://github.com/<user>/<repo>/releases/download/<tag>
                              https://<bucket>.s3.amazonaws.com/tbr
                              https://huggingface.co/<user>/<model-repo>/resolve/main
  TBR_WEIGHTS_TOKEN      optional bearer token, sent as "Authorization: Bearer <token>" (private storage)
  TBR_CHECKPOINT_DIR     where to put them (default: ./checkpoints_best next to this file)

If a file named SHA256SUMS (lines of "<sha256>  <filename>") sits next to the weights it is used to verify every
download. Files that are already present are never downloaded again.

Command line check:  python fetch_weights.py
"""

import hashlib
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

FOLDS = (1, 2, 3, 4, 5)
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_best")


def weight_filename(fold):
    return f"dinov3_vitb16_depth_fold{fold}_best.pt"


def missing_weights(ckpt_dir):
    return [f for f in FOLDS if not os.path.isfile(os.path.join(ckpt_dir, weight_filename(f)))]


class _DropAuthOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """Storage services (Hugging Face, GitHub, S3 ...) redirect downloads to a signed URL on another host. Python would
    forward the Authorization header there, which leaks the token and makes some of them reject the request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_DropAuthOnCrossHostRedirect)


def _request(url, token):
    headers = {"User-Agent": "tbr-depth-app"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _read_sums(base_url, token):
    try:
        with _OPENER.open(_request(base_url.rstrip("/") + "/SHA256SUMS", token), timeout=60) as r:
            sums = {}
            for line in r.read().decode("utf-8", "replace").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    sums[parts[-1].lstrip("*")] = parts[0].lower()
            return sums
    except (urllib.error.URLError, OSError):
        return {}                                     # checksum file is optional


def ensure_weights(ckpt_dir=None, base_url=None, token=None, log=print):
    """Make sure all 5 checkpoints exist in ckpt_dir, downloading the missing ones. Returns the folds downloaded."""
    ckpt_dir = ckpt_dir or os.environ.get("TBR_CHECKPOINT_DIR") or DEFAULT_DIR
    base_url = base_url or os.environ.get("TBR_WEIGHTS_BASE_URL")
    token = token or os.environ.get("TBR_WEIGHTS_TOKEN")
    missing = missing_weights(ckpt_dir)
    if not missing:
        return []
    if not base_url:
        raise FileNotFoundError(
            f"Model weights missing in '{ckpt_dir}' (fold(s) {missing}). Either copy the 5 files "
            f"{weight_filename(1)} ... {weight_filename(5)} into that folder, or set TBR_WEIGHTS_BASE_URL so they "
            f"are downloaded automatically.")
    os.makedirs(ckpt_dir, exist_ok=True)
    sums = _read_sums(base_url, token)
    done = []
    for f in missing:
        name = weight_filename(f)
        url = base_url.rstrip("/") + "/" + name
        dest = os.path.join(ckpt_dir, name)
        part = dest + ".part"
        log(f"Downloading {name} ...")
        h = hashlib.sha256()
        try:
            with _OPENER.open(_request(url, token), timeout=120) as r, open(part, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
        except urllib.error.HTTPError as e:
            if os.path.exists(part):
                os.remove(part)
            raise RuntimeError(f"Could not download {url} (HTTP {e.code}). Check TBR_WEIGHTS_BASE_URL"
                               f"{' / TBR_WEIGHTS_TOKEN' if e.code in (401, 403) else ''}.") from e
        except (urllib.error.URLError, OSError) as e:
            if os.path.exists(part):
                os.remove(part)
            raise RuntimeError(f"Could not download {url}: {e}") from e
        expected = sums.get(name)
        if expected and h.hexdigest().lower() != expected:
            os.remove(part)
            raise RuntimeError(f"Checksum mismatch for {name} (expected {expected[:12]}..., got {h.hexdigest()[:12]}...). "
                               f"The download is corrupted or the file was replaced.")
        os.replace(part, dest)                        # atomic: a half-downloaded file is never mistaken for a weight file
        done.append(f)
    return done


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    got = ensure_weights(folder)
    print("Downloaded folds:", got if got else "none needed -- all 5 weight files present")

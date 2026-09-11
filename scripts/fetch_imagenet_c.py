"""Stream the official ImageNet-C tarballs, keep only the 100 subset classes.

The Zenodo tarballs are large (noise 22.6 GB, blur 7.1, weather 12.8,
digital 7.8). We never write them to disk: each is streamed and untarred on
the fly, and only members whose wnid is in data/subset_classes.json are
extracted to data/imagenet-c/<corruption>/<severity>/<wnid>/. Kept on disk is
~10% of full ImageNet-C, ~5 GB for the 15 standard corruptions.

The HTTP reader transparently reconnects with a Range request if the
connection drops mid-stream, so a flaky link doesn't restart the download.
A tarball that finishes gets a data/imagenet-c/.done_<name> marker and is
skipped on re-run.
"""
import argparse
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "https://zenodo.org/records/2235448/files"

#: Zenodo tarball name -> the corruptions it contains.
TARBALLS = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "extra":   ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
}
#: the four tarballs holding the 15 standard corruptions
STANDARD = ["noise", "blur", "weather", "digital"]

WNID_RE = re.compile(r"n\d{8}")
DATA_DIR = Path("data/imagenet-c")


class ResumableHTTPReader:
    """A read-only file object over an HTTP body that survives dropped connections.

    `tarfile` reads the gzip stream sequentially through `read()`. If the socket
    drops (an exception, or a short empty read that http.client returns without
    raising), this reconnects with a `Range: bytes=<pos>-` request and continues
    from the exact byte offset, so the download resumes instead of restarting.
    """

    def __init__(self, url, max_retries=100, backoff=5.0):
        """Open `url` and prepare to stream it.

        Args:
            url: the file to stream.
            max_retries: give up after this many reconnects.
            backoff: seconds to wait before reconnect n (grows linearly, capped
                at 60).
        """
        self.url = url
        self.max_retries = max_retries
        self.backoff = backoff
        self.tries = 0
        self.pos = 0
        self.length = None
        self._open()

    def _reconnect(self, why):
        """Close the current response and reopen from `self.pos`; raise if over budget."""
        self.tries += 1
        if self.tries > self.max_retries:
            raise RuntimeError(f"gave up after {self.tries} reconnects ({why})")
        wait = min(self.backoff * self.tries, 60.0)
        print(f"\n  [stream drop at {self.pos/1e9:.2f} GB: {why}; "
              f"reconnect {self.tries}/{self.max_retries} in {wait:.0f}s]", flush=True)
        time.sleep(wait)
        try:
            self.resp.close()
        except Exception:
            pass
        self._open()

    def _open(self):
        """(Re)issue the HTTP request, with a Range header once `self.pos` > 0."""
        req = urllib.request.Request(self.url)
        if self.pos:
            req.add_header("Range", f"bytes={self.pos}-")
        self.resp = urllib.request.urlopen(req, timeout=60)
        # the tar/gzip stream can only be resumed if the server gives us exactly
        # the bytes from self.pos on; a 200 here would restart the body and
        # silently corrupt the decode.
        if self.pos and getattr(self.resp, "status", None) != 206:
            raise RuntimeError(
                f"resume needs HTTP 206 but got {getattr(self.resp, 'status', '?')}"
            )
        if self.length is None:
            cl = self.resp.headers.get("Content-Length")
            self.length = int(cl) + self.pos if cl else None

    def read(self, n):
        """Return up to `n` bytes, reconnecting as needed. `b''` only at true EOF."""
        while True:
            try:
                chunk = self.resp.read(n)
            except Exception as e:  # noqa: BLE001 - any socket/http error -> retry
                self._reconnect(repr(e))
                continue
            if chunk:
                self.pos += len(chunk)
                self.tries = 0  # made progress, reset the retry budget
                return chunk
            # empty read: real EOF only if we've reached the known length.
            # http.client returns b'' (no exception) on a dropped socket, and
            # tarfile would take that as "stream ended" -> ReadError. Treat a
            # short empty read as a drop and reconnect.
            if self.length is not None and self.pos >= self.length:
                return b""
            self._reconnect(f"premature EOF ({self.pos} / {self.length})")

    def close(self):
        """Close the underlying HTTP response (errors ignored)."""
        try:
            self.resp.close()
        except Exception:
            pass


def fetch_tarball(name, corruptions, want_wnids, severities, dry_run=False):
    """Stream one Zenodo tarball and extract the images we want.

    Args:
        name: tarball key (e.g. "noise"); see `TARBALLS`.
        corruptions: set of corruption names to keep from this tarball.
        want_wnids: set of wnids (the subset classes) to keep.
        severities: set of severity strings to keep, or None for all.
        dry_run: count matches but write nothing.

    Skips entirely if `data/imagenet-c/.done_<name>` exists; writes that marker
    on success. Files already on disk at the right size are not re-extracted.
    """
    marker = DATA_DIR / f".done_{name}"
    if marker.exists():
        print(f"[{name}] already done (marker present), skipping")
        return
    url = f"{BASE_URL}/{name}.tar?download=1"
    print(f"[{name}] streaming {url}")
    reader = ResumableHTTPReader(url)
    kept = skipped = 0
    seen_keys = set()
    t0 = time.time()
    with tarfile.open(fileobj=reader, mode="r|*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            if parts and parts[0] == DATA_DIR.name:
                parts = parts[1:]
            if len(parts) < 4:
                continue
            corruption, severity, wnid, fname = parts[-4], parts[-3], parts[-2], parts[-1]
            if corruption not in corruptions:
                skipped += 1
                continue
            if severities and severity not in severities:
                skipped += 1
                continue
            if wnid not in want_wnids:
                skipped += 1
                continue

            key = (corruption, severity)
            if key not in seen_keys:
                seen_keys.add(key)
                rate = reader.pos / 1e9 / max(time.time() - t0, 1e-6)
                print(f"  {corruption}/{severity}  "
                      f"(kept {kept}, read {reader.pos/1e9:.1f} GB, {rate:.1f} GB/min*)"
                      .replace("GB/min", "GB/s"))
            if dry_run:
                kept += 1
                continue
            dest_dir = DATA_DIR / corruption / severity / wnid
            dest = dest_dir / fname
            if dest.exists() and dest.stat().st_size == m.size:
                skipped += 1  # already have it (idempotent re-run)
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(dest, "wb") as out:
                out.write(src.read())
            kept += 1
    reader.close()
    dt = time.time() - t0
    print(f"[{name}] done: kept {kept} files, {reader.pos/1e9:.1f} GB streamed, "
          f"{dt/60:.1f} min")
    if not dry_run:
        marker.write_text(f"kept={kept}\nstreamed_bytes={reader.pos}\n")


def main():
    """Parse args and stream each requested tarball in turn."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/subset_classes.json")
    ap.add_argument("--tarballs", nargs="+", default=STANDARD,
                    choices=list(TARBALLS) + ["all"],
                    help="which Zenodo tarballs to stream (default: the 15 standard)")
    ap.add_argument("--corruptions", nargs="+", default=None,
                    help="restrict to these corruption names (default: all in the tarballs)")
    ap.add_argument("--severities", nargs="+", default=None,
                    help="restrict to these severities, e.g. 1 3 5 (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stream and count but write nothing")
    args = ap.parse_args()

    want_wnids = set(json.load(open(args.subset))["wnid_to_index"])
    print(f"subset: {len(want_wnids)} classes")

    tbs = list(TARBALLS) if "all" in args.tarballs else args.tarballs
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in tbs:
        corruptions = TARBALLS[name]
        if args.corruptions:
            corruptions = [c for c in corruptions if c in args.corruptions]
            if not corruptions:
                continue
        fetch_tarball(name, set(corruptions), want_wnids,
                      set(args.severities) if args.severities else None,
                      dry_run=args.dry_run)

    print("\nall requested tarballs processed.")


if __name__ == "__main__":
    main()

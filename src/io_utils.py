"""Download helpers for Maxar Open Data ARD assets."""
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # use the Windows certificate store
import requests

ROOT = Path(__file__).resolve().parents[1]
TILE_DIR = ROOT / "data/raw/maxar_tiles"


def local_path(url: str) -> Path:
    # .../ard/<zone>/<quadkey>/<date>/<catalog>-<asset>.tif
    parts = url.split("/ard/")[1].split("/")
    return TILE_DIR.joinpath(*parts)


def fetch(url: str, chunk=1 << 20) -> Path:
    out = local_path(url)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for b in r.iter_content(chunk):
                fh.write(b)
    tmp.replace(out)
    return out

"""Download OpenStreetMap roads within the study extent (Overpass API).
OSM is used only to locate candidate road segments for the controlled
dislocation experiment, never as positional reference."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import io_utils  # noqa: F401  (injects system trust store)
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/osm/roads_aoi.json"
S, W, N, E = 37.40, 36.95, 37.65, 37.40
q = f"""[out:json][timeout:180];
way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|track|service)$"]({S},{W},{N},{E});
out tags geom;"""
OUT.parent.mkdir(parents=True, exist_ok=True)
r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=300,
                  headers={"User-Agent": "road-offset-pansharpening-study/1.0",
                           "Accept": "application/json"})
r.raise_for_status()
OUT.write_text(r.text, encoding="utf-8")
print(len(json.loads(r.text)["elements"]), "ways")

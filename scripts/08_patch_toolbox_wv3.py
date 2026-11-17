"""Create patched copies of toolbox v1.3 functions with a 'WV3' sensor case
(Nyquist MTF gains of the 2021 benchmark toolbox: MS bands
[0.325 0.355 0.360 0.350 0.365 0.360 0.335 0.315], PAN 0.14).
The copies live in matlab/toolbox_patch and are placed first on the path."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "external/vivone_toolbox/pansharpening_tool_v1.3"
OUT = ROOT / "matlab/toolbox_patch"
OUT.mkdir(parents=True, exist_ok=True)
MS = "    case 'WV3'\n        GNyq = [0.325 0.355 0.360 0.350 0.365 0.360 0.335 0.315];\n"
for rel in ["Quality_Indices/MTF.m", "GLP/MTF_GLP_HPM.m", "resize_images.m", "Quality_Indices/MTF_PAN.m"]:
    txt = (TB / rel).read_text(encoding="latin-1")
    if rel.endswith("MTF_PAN.m"):
        new = re.sub(r"(\n\s*case 'none')", "\n    case 'WV3'\n        GNyq = 0.14;\\1", txt, count=1)
    elif rel == "resize_images.m":
        add = ("    case 'WV3'\n        flag_resize_new = 2; % MTF usage\n"
               "        GNyq = [0.325 0.355 0.360 0.350 0.365 0.360 0.335 0.315];\n        GNyqPan = 0.14;\n")
        new = re.sub(r"(\n\s*case 'none')", "\n" + add.rstrip("\n") + r"\1", txt, count=1)
    else:
        new = re.sub(r"(\n\s*case 'none')", "\n" + MS.rstrip("\n") + r"\1", txt, count=1)
    assert new != txt, rel
    (OUT / Path(rel).name).write_text(new, encoding="latin-1")
    print("patched", rel)

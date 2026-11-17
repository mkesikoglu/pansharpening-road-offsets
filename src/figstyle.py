"""Shared figure style (journal print: Elsevier single column 90 mm, double 190 mm)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
COL1, COL2 = 90 * MM, 165 * MM  # COL2 = text width of the manuscript
PRODUCTS = ["PAN", "EXP", "GSA", "BDSD_PC", "MTF_GLP_HPM", "AWLP", "ZPNN", "VIS"]
LABEL = {"PAN": "PAN", "EXP": "EXP", "GSA": "GSA", "BDSD_PC": "BDSD-PC", "MTF_GLP_HPM": "MTF-GLP-HPM",
         "AWLP": "AWLP", "ZPNN": "Z-PNN", "VIS": "Vendor RGB"}
# categorical palette in fixed order (validated default palette of the dataviz method)
COLOR = dict(zip(PRODUCTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
                            "#e34948"]))
MARKER = dict(zip(PRODUCTS, ["o", "s", "^", "v", "D", "P", "X", "*"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.5, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "mathtext.fontset": "dejavusans"})


def panel_label(ax, s, x=-0.02, y=1.02):
    """Panel letter. If the axes already has a title, the letter is merged into it
    (left-aligned) so that letter and title cannot overlap."""
    title = ax.get_title()
    if title:
        fsz = ax.title.get_fontsize()
        ax.set_title("")
        ax.set_title(r"$\mathbf{" + s.replace("(", r"(").replace(")", r")") + "}$  " + title, loc="left", fontsize=fsz)
        return
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=9.5, ha="right", va="bottom",
            color=INK)


def halo(txt, lw=2.5):
    """White outline behind a text artist, for labels drawn over data."""
    import matplotlib.patheffects as pe
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])
    return txt

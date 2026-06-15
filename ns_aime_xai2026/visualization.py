"""
ns_aime_xai2026.visualization
======================

Static, "catchy" AIME-style visualizations for NS-AIME explanations.

These figures deliberately mirror the visual language of the AIME inverse-
operator views (cream paper, coral = promote, indigo = inhibit, rounded nodes,
A-dagger branding) rather than the plain academic graph in the paper.  The
*direction* of every edge is derived automatically from the induced rules:

* edges always flow  feature -> class  (and feature -> AND -> class), because
  the explanation answers "which inputs drive this output";
* the *sign / role* (promote vs. inhibit) is taken from the induced monotonicity
  and rendered as colour (coral / indigo), so the direction of influence is
  read at a glance.

Pure-matplotlib implementation (no networkx dependency).
"""

from __future__ import annotations

from typing import Optional, Sequence, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.patheffects as pe


# --- AIME palette ---------------------------------------------------------- #
INK = "#1B2A4A"
SOFT = "#5C6680"
PAPER = "#FBF9F4"
INDIGO = "#21307A"     # negative / inhibit
CORAL = "#D2552E"      # positive / promote
GOLD = "#E0A23B"
VIOLET = "#7A4FA0"     # AND nodes
GRIDLINE = "#E6E0D6"


def _sign_color(sign: float) -> str:
    return CORAL if sign > 0 else INDIGO


def _glow_edge(ax, p0, p1, color, width, rad, alpha=1.0, zorder=2):
    """Draw a curved arrow feature->class with a soft glow underlay."""
    # glow underlay (a few translucent strokes)
    for w, a in ((width + 6, 0.06), (width + 3, 0.10)):
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", linewidth=w, color=color, alpha=a,
            shrinkA=14, shrinkB=22, zorder=zorder - 0.1))
    # crisp arrow on top
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=16 + width,
        linewidth=width, color=color, alpha=alpha,
        shrinkA=14, shrinkB=22, zorder=zorder))


def _chip(ax, xy, text, fg=INK, bg="#FFFFFF", edge=GRIDLINE, fontsize=8.5):
    t = ax.text(xy[0], xy[1], text, ha="center", va="center",
                fontsize=fontsize, color=fg, zorder=6, fontweight="bold")
    t.set_bbox(dict(boxstyle="round,pad=0.3", fc=bg, ec=edge, lw=1.0))
    return t


def _node(ax, xy, r, fc, label, label_color="#FFFFFF", fontsize=10,
          ring=PAPER, ring_w=2.0, square=False, zorder=5):
    if square:
        patch = FancyBboxPatch(
            (xy[0] - r, xy[1] - r), 2 * r, 2 * r,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fc, ec=ring, lw=ring_w, zorder=zorder)
    else:
        patch = Circle(xy, r, fc=fc, ec=ring, lw=ring_w, zorder=zorder)
    patch.set_path_effects([pe.withSimplePatchShadow(
        offset=(2, -2), alpha=0.18)])
    ax.add_patch(patch)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=label_color,
            zorder=zorder + 1)


def plot_logic_graph(
    B: np.ndarray,
    rules: List[List[Dict[str, Any]]],
    feature_names: Sequence[str],
    class_index: int,
    class_name: Optional[str] = None,
    top_k: int = 10,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    title: Optional[str] = None,
):
    """Plot a catchy radial logic graph for one class.

    Promoting features (positive monotonicity) fan out on the upper arc in
    coral; inhibiting features on the lower arc in indigo.  Pairwise AND rules
    appear as rounded violet nodes feeding the class.  Edge thickness encodes
    ``|B[j, k]|`` and each feature carries a threshold chip (``>= t`` / ``<= t``
    / ``present``).
    """
    if class_name is None:
        class_name = f"class {class_index}"

    coefs = np.asarray(B)[:, class_index]
    class_rules = rules[class_index]
    single = {r["feature_idx"]: r for r in class_rules if r.get("type") == "single"}

    # choose features to show (by |B|), skip ~zero
    order = np.argsort(np.abs(coefs))[::-1]
    feats = [int(j) for j in order if abs(coefs[j]) > 1e-6][:top_k]
    if not feats:
        feats = [int(order[0])]

    # split by induced sign (rule first, else weight sign) -> top / bottom arcs
    def role(j):
        if j in single:
            return single[j]["sign"]
        return 1.0 if coefs[j] >= 0 else -1.0

    pos = [j for j in feats if role(j) > 0]
    neg = [j for j in feats if role(j) <= 0]
    wmax = max(abs(coefs[j]) for j in feats) or 1.0

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(12.5, 9))
    else:
        fig = ax.figure
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    hub = np.array([0.0, 0.0])
    R = 1.0  # feature ring radius

    def arc_positions(items, top):
        """Place items on an upper (top=True) or lower arc."""
        out = {}
        if not items:
            return out
        if top:
            angs = np.linspace(np.deg2rad(35), np.deg2rad(145), len(items))
        else:
            angs = np.linspace(np.deg2rad(-35), np.deg2rad(-145), len(items))
        for j, a in zip(items, angs):
            out[j] = hub + R * np.array([np.cos(a), np.sin(a)])
        return out

    pos_xy = {}
    pos_xy.update(arc_positions(pos, top=True))
    pos_xy.update(arc_positions(neg, top=False))

    # --- edges: feature -> class ----------------------------------------- #
    for j in feats:
        w = coefs[j]
        sgn = role(j)
        col = _sign_color(sgn)
        lw = 1.6 + 5.5 * (abs(w) / wmax)
        p = pos_xy[j]
        rad = 0.18 if p[1] >= 0 else -0.18
        _glow_edge(ax, p, hub, col, lw, rad)

    # --- AND rule nodes -------------------------------------------------- #
    and_rules = [r for r in class_rules if r.get("type") == "and"]
    and_rules = sorted(and_rules, key=lambda r: r["importance"], reverse=True)[:3]
    for i, r in enumerate(and_rules):
        ang = np.deg2rad(180 + (i - (len(and_rules) - 1) / 2) * 26)
        ap = hub + 0.55 * np.array([np.cos(ang), np.sin(ang)])
        # feature -> AND (only if features are on the graph)
        for jj in r["features"]:
            if jj in pos_xy:
                _glow_edge(ax, pos_xy[jj], ap, VIOLET, 2.2, 0.05, alpha=0.9)
        _glow_edge(ax, ap, hub, VIOLET, 3.0, 0.0, alpha=0.95)
        _node(ax, ap, 0.085, VIOLET, "&", square=True, fontsize=13)

    # --- feature nodes + threshold chips --------------------------------- #
    for j in feats:
        p = pos_xy[j]
        sgn = role(j)
        size = 0.085 + 0.05 * (abs(coefs[j]) / wmax)
        _node(ax, p, size, "#FFFFFF", "", ring=_sign_color(sgn), ring_w=2.6)
        # feature label outside the node
        out = p + 0.16 * p / (np.linalg.norm(p) + 1e-9)
        ha = "left" if out[0] > 0.02 else ("right" if out[0] < -0.02 else "center")
        ax.text(out[0], out[1] + (0.04 if out[1] >= 0 else -0.04),
                feature_names[j], ha=ha, va="center", fontsize=10.5,
                color=INK, fontweight="bold", zorder=6)
        # threshold chip
        r = single.get(j)
        if r is not None:
            if r["presence"]:
                txt = "present"
            else:
                op = "≥" if r["sign"] > 0 else "≤"
                td = r.get("threshold_display", r["threshold"])
                txt = f"{op} {td:.2f}" if abs(td) >= 0.005 else f"{op} 0"
            chip_xy = hub + 0.62 * (p - hub)
            _chip(ax, chip_xy, txt, fg=_sign_color(r["sign"]))

    # --- class hub ------------------------------------------------------- #
    _node(ax, hub, 0.17, INK, class_name, label_color="#FFFFFF",
          fontsize=12.5, ring=GOLD, ring_w=3.0, zorder=8)

    # --- frame, legend, branding ----------------------------------------- #
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.45, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    ttl = title or f"NS-AIME  ·  logic graph"
    ax.text(-1.66, 1.46, ttl, fontsize=17, fontweight="bold", color=INK)
    ax.text(-1.66, 1.33, f"rule-structured explanation for  “{class_name}”",
            fontsize=11, color=SOFT)
    ax.add_patch(FancyBboxPatch((-1.66, 1.27), 0.42, 0.012,
                 boxstyle="round,pad=0.002", fc=CORAL, ec="none"))

    # legend
    lg = [(CORAL, "promotes (M = +1)"), (INDIGO, "inhibits (M = −1)"),
          (VIOLET, "pairwise AND")]
    for i, (c, lab) in enumerate(lg):
        y = -1.34 - 0.0
        x = -1.6 + i * 0.95
        ax.add_patch(Circle((x, y), 0.028, fc=c, ec="none"))
        ax.text(x + 0.06, y, lab, fontsize=9.5, color=SOFT, va="center")

    ax.text(1.66, -1.4, "NS-AIME · A† → rules", ha="right",
            fontsize=10, fontweight="bold", color=INDIGO, alpha=0.85)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150,
                    facecolor=PAPER)
    if show and created:
        plt.show()
    return ax


def plot_dual_report(
    A_dagger: np.ndarray,
    B: np.ndarray,
    feature_names: Sequence[str],
    class_index: int,
    class_name: Optional[str] = None,
    top_k: int = 12,
    rd: Optional[float] = None,
    ga: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Dual-reporting bar chart (Section 3.4 / Fig. 2).

    Raw AIME (indigo) and NS-AIME (coral) attributions are shown side by side so
    that large divergences - the audit signal - are visible.  RD / GA can be
    annotated in the corner.
    """
    if class_name is None:
        class_name = f"class {class_index}"
    a = np.asarray(A_dagger)[:, class_index]
    b = np.asarray(B)[:, class_index]
    idx = np.argsort(np.abs(a) + np.abs(b))[::-1][:top_k]
    names = [feature_names[j] for j in idx]
    y = np.arange(len(idx))[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(idx) + 1.8))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    h = 0.38
    ax.barh(y + h / 2, a[idx], height=h, color=INDIGO, alpha=0.85,
            label="raw AIME  (A†)")
    ax.barh(y - h / 2, b[idx], height=h, color=CORAL, alpha=0.9,
            label="NS-AIME  (B)")
    ax.axvline(0, color=SOFT, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlabel("contribution to  “" + class_name + "”", color=SOFT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDLINE)
    ax.tick_params(colors=SOFT)
    ax.set_title("Dual reporting  ·  raw AIME vs NS-AIME",
                 fontsize=14, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    if rd is not None or ga is not None:
        msg = []
        if rd is not None:
            msg.append(f"RD = {rd:.2f}")
        if ga is not None:
            msg.append(f"GA = {ga:+.2f}")
        flag = (ga is not None and ga < 0.2) or (rd is not None and rd > 3.0)
        box = CORAL if flag else INDIGO
        note = "  ⚠ audit signal" if flag else ""
        ax.text(0.99, 1.02, "   ".join(msg) + note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10.5, fontweight="bold",
                color=box)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=PAPER)
    if show:
        plt.show()
    return ax

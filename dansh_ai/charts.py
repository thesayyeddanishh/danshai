"""Matplotlib charts in the dashboard's look (orange #ff5000, royalblue boundaries, red wickets, grey others).

Orientation notes (same data conventions as the pages):
  * pitch map: x = BounceY (bowler's view, + = bowler's right), y = BounceX with batter's end at the top
  * beehive: x = CreaseY, y = CreaseZ, zone shading from cricket_config
  * wagon wheel: BOWLER'S VIEW - bowler at the bottom, 'behind the stumps' at the top, x = LandingY
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge

from cricket_config import add_crease_lateral_zone_background, get_pitch_bins, pitch_bin_percentages
from .prep import Prepared

ORANGE, BLUE, GREY, RED = "#ff5000", "royalblue", "#D3D3D3", "red"


def _empty(msg: str, size=(5, 3)):
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
    ax.axis("off")
    plt.close(fig)
    return fig


def _split(df: pd.DataFrame):
    wk = df[df["IsWicket"]]
    nw = df[~df["IsWicket"]]
    bd = nw[nw["Boundary"]]
    return nw[~nw["Boundary"]], bd, wk


def pitch_map(df: pd.DataFrame, prep: Prepared, title: str = "") -> plt.Figure:
    d = df[df["ok_bounce"]]
    if d.empty:
        return _empty("No tracked deliveries for the pitch map")
    cls = d["DeliveryClass"].mode().iat[0]
    bins = get_pitch_bins("Seam" if cls == "Medium" else cls, prep.cfg)
    fig, ax = plt.subplots(figsize=(3.5, 7))
    for lo, _hi in bins.values():
        if lo > -4.0:
            ax.axhline(lo, color="lightgrey", lw=1.0, ls="--")
    pct = pitch_bin_percentages(d, bins, "BounceX")
    for name, (lo, hi) in bins.items():
        mid = (lo + hi) / 2
        ax.text(-1.45, mid + 0.2, name.upper(), fontsize=8, fontweight="bold", va="center")
        ax.text(-1.45, mid - 0.2, f"{int(pct.get(name, 0))}%", fontsize=16, fontweight="bold", va="center")
    oth, bd, wk = _split(d)
    for part, c, s, a, lab in ((oth, GREY, 100, 0.9, "Others"), (bd, BLUE, 100, 0.75, "Boundaries"), (wk, RED, 150, 1.0, "Wicket")):
        if not part.empty:
            ax.scatter(part["BounceY"], part["BounceX"], s=s, c=c, edgecolor="white", linewidths=0.6, alpha=a, label=lab)
    for x in (-0.115, 0, 0.115):
        ax.axvline(x, color="#777777", lw=0.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(16.0, -4.0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\n{len(d)} tracked balls", fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def beehive(df: pd.DataFrame, prep: Prepared, title: str = "") -> plt.Figure:
    d = df[df["ok_crease"]]
    if d.empty:
        return _empty("No tracked deliveries for the beehive")
    hands = [h for h in (True, False) if (d["BatterRH"] == h).any()]
    fig, axes = plt.subplots(1, len(hands), figsize=(4.2 * len(hands) + 1.5, 4.6), squeeze=False)
    for ax, rh in zip(axes[0], hands):
        part = d[d["BatterRH"] == rh]
        add_crease_lateral_zone_background(ax, rh)
        oth, bd, wk = _split(part)
        for p_, c, s, a in ((oth, GREY, 40, 0.9), (bd, BLUE, 45, 0.8), (wk, RED, 70, 1.0)):
            if not p_.empty:
                ax.scatter(p_["CreaseY"], p_["CreaseZ"], s=s, c=c, edgecolor="white", linewidths=0.4, alpha=a)
        ax.add_patch(Rectangle((-0.115, 0), 0.23, 0.746, fill=False, ec="#444444", lw=1))
        zones = part.groupby("LateralZone").agg(n=("Runs", "size"), w=("IsWicket", "sum"), r=("Runs", "sum"))
        txt = " | ".join(f"{z.title()}: {int(r.n)}b {int(r.w)}w" for z, r in zones.reindex(["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]).dropna().iterrows())
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 2.0); ax.set_xlabel("CreaseY (m, bowler's view)", fontsize=8); ax.set_ylabel("CreaseZ (m)", fontsize=8)
        ax.set_title(f"{'Right' if rh else 'Left'}-handers - {len(part)} balls", fontsize=9)
        ax.text(0.5, -0.22, txt, transform=ax.transAxes, ha="center", fontsize=6.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.close(fig)
    return fig


# wagon sectors in bowler's-view plot coordinates (x = LandingY, y = -LandingX), degrees CCW from +x
_SECT = {
    True: {"FINE LEG": (0, 90), "THIRD MAN": (90, 180), "COVER": (180, 225), "LONG OFF": (225, 270), "LONG ON": (270, 315), "SQUARE LEG": (315, 360)},
    False: {"THIRD MAN": (0, 90), "FINE LEG": (90, 180), "SQUARE LEG": (180, 225), "LONG ON": (225, 270), "LONG OFF": (270, 315), "COVER": (315, 360)},
}


def wagon_wheel(df: pd.DataFrame, prep: Prepared, title: str = "") -> plt.Figure:
    d = df[df["Wagon"].notna()]
    if d.empty:
        return _empty("No scoring shots with landing data")
    hands = [h for h in (True, False) if (d["BatterRH"] == h).any()]
    fig, axes = plt.subplots(1, len(hands), figsize=(4.6 * len(hands), 4.8), squeeze=False)
    for ax, rh in zip(axes[0], hands):
        part = d[d["BatterRH"] == rh]
        runs = part.groupby("Wagon")["Runs"].sum()
        total = runs.sum()
        top = runs.idxmax() if len(runs) else None
        for name, (a0, a1) in _SECT[rh].items():
            r = float(runs.get(name, 0))
            ax.add_patch(Wedge((0, 0), 1, a0, a1, fc=ORANGE if name == top else "white", ec="black", lw=0.8))
            mid = np.deg2rad((a0 + a1) / 2)
            col = "white" if name == top else "black"
            ax.text(0.62 * np.cos(mid), 0.62 * np.sin(mid), f"{name}\n{100 * r / total:.0f}%\n{int(r)} runs", ha="center", va="center", fontsize=7, color=col, fontweight="bold")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{'Right' if rh else 'Left'}-handers - {int(total)} runs", fontsize=9)
    fig.suptitle(f"{title}\n(bowler's view: bowler at bottom)", fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def length_bars(df: pd.DataFrame, prep: Prepared, role: str, title: str = "") -> plt.Figure:
    d = df[df["LengthBin"].notna()]
    if d.empty:
        return _empty("No tracked deliveries for length")
    from .metrics import group_table
    t, _ = group_table(d, prep, role, ["length"])
    key = "SR" if role == "batter" else ("BowlSR" if prep.cfg.is_test else "Econ")
    fig, ax = plt.subplots(figsize=(5, 0.7 * len(t) + 1.6))
    pct = t["Balls"] / t["Balls"].sum() * 100
    y = np.arange(len(t))[::-1]
    ax.barh(y, pct.values, color=ORANGE, alpha=0.9, height=0.55)
    for yi, (idx, row) in zip(y, t.iterrows()):
        v = row[key]
        ax.text(pct.max() * 1.02, yi, f"{int(row['Balls'])}b  {key} {'-' if pd.isna(v) else f'{v:.1f}'}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(list(t.index), fontsize=9)
    ax.set_xlim(0, pct.max() * 1.6); ax.set_xlabel("% of balls", fontsize=8)
    ax.set_title(title or "Length", fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def speed_hist(df: pd.DataFrame, prep: Prepared, title: str = "") -> plt.Figure:
    s = df["SpeedV"].dropna()
    if s.empty:
        return _empty("No tracked speeds")
    lo, hi = (50, 155) if prep.unit == "kph" else (30, 100)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(s.clip(lo, hi), bins=np.arange(lo, hi + 5, 5), color=ORANGE, alpha=0.9, edgecolor="white")
    ax.axvline(s.mean(), color="black", ls="--", lw=1)
    ax.set_xlabel(f"Release speed ({prep.unit})", fontsize=8); ax.set_ylabel("Balls", fontsize=8)
    ax.set_title(f"{title}  (mean {s.mean():.1f}, n={len(s)})", fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def bar_chart(table: pd.DataFrame, value_col: str, title: str = "", label_fmt: str = "{:.1f}") -> plt.Figure:
    t = table.dropna(subset=[value_col])
    if t.empty:
        return _empty("Nothing to plot")
    labels = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index]
    fig, ax = plt.subplots(figsize=(5.5, 0.45 * len(t) + 1.6))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t[value_col].values, color=ORANGE, alpha=0.9, height=0.6)
    for yi, (v, n) in zip(y, zip(t[value_col].values, t["Balls"].values)):
        ax.text(v, yi, f" {label_fmt.format(v)}  ({int(n)}b)", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, t[value_col].max() * 1.35)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def trend(table: pd.DataFrame, value_col: str, title: str = "") -> plt.Figure:
    t = table.dropna(subset=[value_col])
    if len(t) < 2:
        return _empty("Need at least 2 points for a trend")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot([str(i) for i in t.index], t[value_col].values, marker="o", color=ORANGE, lw=2)
    for x, v, n in zip([str(i) for i in t.index], t[value_col].values, t["Balls"].values):
        ax.annotate(f"{v:.1f}\n({int(n)}b)", (x, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.margins(y=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig

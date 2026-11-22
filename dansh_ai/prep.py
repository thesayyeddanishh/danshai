"""Data preparation: builds a DERIVED, analysis-ready copy of the uploaded CSV.

Rules implemented here come from the DANsh Q&A + the data checks on LPL.xlsx:
  * names are case/space-normalised (no fuzzy identity merging), handedness resolved by majority
  * Medium counts as Seam; delivery classes are Seam / Spin
  * untracked balls (ReleaseSpeed < 0, all trajectory fields 0) stay in runs/balls/wickets but are
    excluded from every trajectory metric; other outliers are excluded PER METRIC only
  * Over and Ball start at 1; wides/no-balls are rows (every row counts as a ball, like the 4 pages)
  * line axis and Swing/Deviation signs are absolute from the bowler's view (+ = bowler's right)
`parity_mode=True` switches the cleaning off so numbers equal the 4 existing pages exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cricket_config import (
    FormatConfig,
    apply_match_phase_filter,
    get_pitch_bins,
    hitting_stumps_mask,
    match_phase_options,
    resolve_format,
    seam_speed_group,
    spin_speed_group,
)
from . import config as C

REQUIRED_COLS = [
    "Innings", "Over", "Ball", "BatsmanName", "BowlerName", "BowlingTeam", "BattingTeam",
    "IsBowlerRightHanded", "IsBatsmanRightHanded", "Wicket", "Runs", "ReleaseY", "ReleaseZ",
    "ReleaseSpeed", "Deviation", "Swing", "DeliveryType", "StumpsY", "StumpsZ", "CreaseY", "CreaseZ",
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", "InterceptionZ", "InterceptionY",
    "Ground", "Tour", "Year", "Match",
]
NUMERIC_COLS = [
    "Innings", "Over", "Ball", "Runs", "Year", "ReleaseY", "ReleaseZ", "ReleaseSpeed", "Deviation", "Swing",
    "StumpsY", "StumpsZ", "CreaseY", "CreaseZ", "LandingX", "LandingY", "BounceX", "BounceY",
    "InterceptionX", "InterceptionY", "InterceptionZ",
]


class MissingColumns(ValueError):
    pass


def norm_name(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).upper()


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    m = s.astype(str).str.strip().str.lower().map({
        "true": True, "t": True, "1": True, "1.0": True, "yes": True, "y": True,
        "false": False, "f": False, "0": False, "0.0": False, "no": False, "n": False,
    })
    return m.fillna(False).astype(bool)


# ---------------------------------------------------------------------------------------------
# geometry helpers (mirror the dashboard pages exactly)
# ---------------------------------------------------------------------------------------------
def wagon_area(lx, ly, rh, runs) -> np.ndarray:
    """Same sectors as `calculate_scoring_wagon` on the Batters/Pacers/Spinners pages."""
    lx = np.asarray(lx, dtype=float)
    ly = np.asarray(ly, dtype=float)
    rh = np.asarray(rh, dtype=bool)
    runs = np.asarray(runs, dtype=float)
    out = np.full(lx.shape, None, dtype=object)
    ok = (runs > 0) & np.isfinite(lx) & np.isfinite(ly)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(lx > 0, np.arctan(ly / np.where(lx > 0, lx, 1.0)), np.nan)
    q = np.pi / 4
    behind = ok & (lx <= 0)
    front = ok & (lx > 0)
    for hand in (True, False):
        h = rh == hand
        if hand:
            out[behind & h & (ly > 0)] = "FINE LEG"
            out[behind & h & (ly <= 0)] = "THIRD MAN"
            out[front & h & (ly < 0) & (theta < -q)] = "COVER"
            out[front & h & (ly < 0) & ~(theta < -q)] = "LONG OFF"
            out[front & h & (ly >= 0) & (theta >= q)] = "SQUARE LEG"
            out[front & h & (ly >= 0) & ~(theta >= q)] = "LONG ON"
        else:
            out[behind & h & (ly > 0)] = "THIRD MAN"
            out[behind & h & (ly <= 0)] = "FINE LEG"
            out[front & h & (ly < 0) & (theta < -q)] = "SQUARE LEG"
            out[front & h & (ly < 0) & ~(theta < -q)] = "LONG ON"
            out[front & h & (ly >= 0) & (theta >= q)] = "COVER"
            out[front & h & (ly >= 0) & ~(theta >= q)] = "LONG OFF"
    return out


def lateral_zone(y, z, rh) -> np.ndarray:
    """Same zones as `assign_lateral_zone` in the beehive charts."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    rh = np.asarray(rh, dtype=bool)
    out = np.full(y.shape, None, dtype=object)
    ok = np.isfinite(y) & np.isfinite(z) & ~(z > C.CREASE_Z_MAX)
    r = ok & rh
    out[r & (y > 0.115) & (y <= 1.75)] = "LEG"
    out[r & (y > -0.115) & (y <= 0.115)] = "STUMPS"
    out[r & (y > -0.415) & (y <= -0.115)] = "OUTSIDE OFF"
    out[r & (y > -1.245) & (y <= -0.415)] = "WAY OUTSIDE OFF"
    l = ok & ~rh
    out[l & (y >= -1.75) & (y < -0.115)] = "LEG"
    out[l & (y >= -0.115) & (y < 0.115)] = "STUMPS"
    out[l & (y >= 0.115) & (y < 0.415)] = "OUTSIDE OFF"
    out[l & (y >= 0.415) & (y < 1.245)] = "WAY OUTSIDE OFF"
    return out



def _suspect_classes(d: pd.DataFrame, unit: str) -> List[Dict]:
    """Bowlers labelled Seam whose tracked median speed is clearly spin-like (manual labels can be wrong)."""
    lim = 100.0 if unit == "kph" else 62.0
    s = d[(d["DeliveryClass"] == "Seam") & (d["ReleaseSpeed"] > 0) & ~d["Untracked"]]
    g = s.groupby("Bowler")["ReleaseSpeed"].agg(["size", "median"])
    g = g[(g["size"] >= 60) & (g["median"] < lim)]
    return [{"bowler": b, "labelled": "Seam", "median_speed": round(float(r["median"]), 1), "balls": int(r["size"])}
            for b, r in g.sort_values("size", ascending=False).iterrows()]


def _possible_same(d: pd.DataFrame) -> List[Dict]:
    """Informational only (never merged): a one-word name equal to the last word of a longer name, same hand."""
    out: List[Dict] = []
    for col, hand, role in (("Bowler", "BowlerRH", "bowler"), ("Batter", "BatterRH", "batter")):
        t = d.groupby(col).agg(n=("Runs", "size"), rh=(hand, "first"), y0=("Year", "min"), y1=("Year", "max"))
        for nm, r in t.iterrows():
            if " " in nm:
                continue
            for other, r2 in t.iterrows():
                if other != nm and other.split()[-1] == nm and r2["rh"] == r["rh"]:
                    out.append({"role": role, "short_name": nm, "short_years": [int(r.y0), int(r.y1)], "short_balls": int(r.n),
                                "full_name": other, "full_years": [int(r2.y0), int(r2.y1)], "full_balls": int(r2.n),
                                "years_overlap": bool(r.y0 <= r2.y1 and r2.y0 <= r.y1)})
    return out


@dataclass
class Prepared:
    df: pd.DataFrame
    cfg: FormatConfig
    unit: str
    parity_mode: bool
    report: Dict
    bowlers: pd.DataFrame = field(default_factory=pd.DataFrame)
    batters: pd.DataFrame = field(default_factory=pd.DataFrame)


def _majority_hand(names: pd.Series, hand: pd.Series):
    tmp = pd.DataFrame({"n": names.values, "h": hand.astype(float).values})
    g = tmp.groupby("n")["h"].agg(["mean", "nunique", "size"])
    resolved = g["mean"] >= 0.5  # tie -> right-handed
    conflicts = [
        {"name": n, "balls": int(r["size"]), "share_right": round(float(r["mean"]), 3)}
        for n, r in g[g["nunique"] > 1].sort_values("size", ascending=False).iterrows()
    ]
    return names.map(resolved).astype(bool), conflicts


def prepare(df_raw: pd.DataFrame, fmt_key: str = "men_t20i", unit: str = "kph", parity_mode: bool = False,
            fix_suspect_class: bool = False, class_overrides: Optional[Dict[str, str]] = None) -> Prepared:
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        raise MissingColumns(f"Missing required columns: {', '.join(missing)}")
    cfg = resolve_format(fmt_key)
    d = df_raw.copy()  # the uploaded data is never modified
    for c in NUMERIC_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["IsWicket"] = to_bool(d["Wicket"])
    d["Runs"] = d["Runs"].fillna(0)

    # --- identity -------------------------------------------------------------------------------
    d["Batter"] = d["BatsmanName"].map(norm_name)
    d["Bowler"] = d["BowlerName"].map(norm_name)
    d["BatterRH"], bat_conf = _majority_hand(d["Batter"], to_bool(d["IsBatsmanRightHanded"]))
    d["BowlerRH"], bowl_conf = _majority_hand(d["Bowler"], to_bool(d["IsBowlerRightHanded"]))
    d["BatterHand"] = np.where(d["BatterRH"], "Right", "Left")
    d["BowlerHand"] = np.where(d["BowlerRH"], "Right", "Left")
    for c in ("BowlingTeam", "BattingTeam", "Ground", "Tour"):
        d[c] = d[c].map(norm_name)
    d["Match"] = d["Match"].astype(str)
    d["MatchNo"] = pd.to_numeric(d["Match"].str.extract(r"_(?:T20I?|ODI|TEST|T10)_(\d+)_", expand=False), errors="coerce")

    # --- order (file is not sorted): Year, match number, innings, over, ball -------------------
    d = d.sort_values(["Year", "MatchNo", "Match", "Innings", "Over", "Ball"], kind="mergesort").reset_index(drop=True)

    # --- delivery class -------------------------------------------------------------------------
    dt = d["DeliveryType"].astype(str).str.strip().str.title()
    n_medium = int((dt == "Medium").sum())
    d["DeliveryClass"] = dt if parity_mode else dt.replace({"Medium": "Seam"})
    d.loc[~d["DeliveryClass"].isin(["Seam", "Spin", "Medium"]), "DeliveryClass"] = "Other"

    # --- tracking validity ----------------------------------------------------------------------
    traj0 = (
        (d["CreaseY"] == 0) & (d["CreaseZ"] == 0) & (d["StumpsY"] == 0)
        & (d["StumpsZ"] == 0) & (d["ReleaseY"] == 0) & (d["ReleaseZ"] == 0)
    )
    d["Untracked"] = ((d["ReleaseSpeed"] < C.UNTRACKED_SPEED) | traj0).fillna(False)
    ut = ~d["Untracked"] if not parity_mode else pd.Series(True, index=d.index)
    suspects = _suspect_classes(d, unit)
    if fix_suspect_class and not parity_mode:
        for sc in suspects:
            d.loc[d["Bowler"] == sc["bowler"], "DeliveryClass"] = "Spin"
    for nm, cls in (class_overrides or {}).items():
        d.loc[d["Bowler"] == norm_name(nm), "DeliveryClass"] = str(cls).title()

    def rng(col: str) -> pd.Series:
        lo, hi = C.RANGES[col]
        return d[col].between(lo, hi)

    if parity_mode:
        d["ok_length"] = d["BounceX"].notna()
        d["ok_speed"] = d["ReleaseSpeed"].notna()
        d["ok_stumps"] = d["StumpsY"].notna() & d["StumpsZ"].notna()
        d["ok_dev"] = d["Deviation"].notna()
        d["ok_swing"] = d["Swing"].notna()
        d["ok_crease"] = d["CreaseY"].notna() & d["CreaseZ"].notna()
        d["ok_bounce"] = d["BounceX"].notna() & d["BounceY"].notna()
        d["ok_release"] = d["ReleaseY"].notna() & d["ReleaseZ"].notna()
        d["has_contact"] = d["InterceptionX"] > -999
    else:
        d["ok_length"] = ut & rng("BounceX")
        d["ok_bounce"] = d["ok_length"] & rng("BounceY")
        d["ok_speed"] = ut & rng("ReleaseSpeed")
        d["ok_stumps"] = ut & rng("StumpsY") & rng("StumpsZ")
        d["ok_dev"] = ut & rng("Deviation")
        d["ok_swing"] = ut & rng("Swing")
        d["ok_crease"] = ut & rng("CreaseY") & rng("CreaseZ")
        d["ok_release"] = ut & rng("ReleaseY") & rng("ReleaseZ")
        d["has_contact"] = ut & (d["InterceptionX"] > C.NO_CONTACT) & d["InterceptionX"].between(-11.0, 0.5)
    d["ok_landing"] = (d["Runs"] > 0) & d["LandingX"].notna() & d["LandingY"].notna()

    # --- result flags -----------------------------------------------------------------------------
    d["Boundary"] = d["Runs"].isin(C.BOUNDARY_RUNS)
    d["Four"] = d["Runs"] == 4
    d["Six"] = d["Runs"] == 6
    d["Dot"] = d["Runs"] == 0

    # --- phase (taken from cricket_config so it cannot drift from the pages) --------------------------
    d["Phase"] = "All"
    if not cfg.is_test:
        for p in match_phase_options(cfg)[1:]:
            d.loc[apply_match_phase_filter(d, p, cfg).index, "Phase"] = p

    # --- trajectory-derived categories -----------------------------------------------------------------
    d["LengthBin"] = None
    for cls in ("Seam", "Spin"):
        for name, (lo, hi) in get_pitch_bins(cls, cfg).items():
            m = (d["DeliveryClass"] == cls) & d["ok_length"] & (d["BounceX"] >= lo) & (d["BounceX"] < hi)
            d.loc[m, "LengthBin"] = name
    zone = lateral_zone(d["CreaseY"].where(d["ok_crease"]), d["CreaseZ"].where(d["ok_crease"]), d["BatterRH"])
    d["LateralZone"] = zone
    d["SpeedBand"] = None
    for cls, fn in (("Seam", seam_speed_group), ("Spin", spin_speed_group)):
        m = (d["DeliveryClass"] == cls) & d["ok_speed"]
        d.loc[m, "SpeedBand"] = d.loc[m, "ReleaseSpeed"].map(lambda s: fn(s, cfg, unit))
    d["HitsStumps"] = hitting_stumps_mask(d) & d["ok_stumps"]
    d["Wagon"] = wagon_area(d["LandingX"], d["LandingY"], d["BatterRH"], d["Runs"].where(d["ok_landing"], 0))

    # over / round the wicket: right-arm releases at ReleaseY < 0 when over; left-arm at ReleaseY > 0
    ry = d["ReleaseY"].where(d["ok_release"])
    over_w = np.where(d["BowlerRH"], ry < 0, ry > 0)
    d["OverRound"] = np.where(ry.isna() | (ry == 0), None, np.where(over_w, "Over the wicket", "Round the wicket"))

    # --- masked numeric columns used for averages ------------------------------------------------------
    d["LenV"] = d["BounceX"].where(d["ok_length"])
    d["SpeedV"] = d["ReleaseSpeed"].where(d["ok_speed"])
    d["SwingV"] = d["Swing"].where(d["ok_swing"])
    d["DevV"] = d["Deviation"].where(d["ok_dev"])
    d["AbsSwingV"] = d["SwingV"].abs()
    d["AbsDevV"] = d["DevV"].abs()
    d["SwingToLegV"] = d["SwingV"] * np.where(d["BatterRH"], 1.0, -1.0)   # + = in-swing (toward batter's leg side)
    d["HitV"] = d["HitsStumps"].astype(float).where(d["ok_stumps"])
    d["ContactDistV"] = (d["InterceptionX"] + 10).where(d["has_contact"])  # metres in front of stumps (+10 as pages)
    d["TurnDir"] = np.where(
        d["DevV"].isna(), None,
        np.where(d["DevV"] < -C.TURN_EPS, "Turn Left", np.where(d["DevV"] > C.TURN_EPS, "Turn Right", "No Turn")),
    )

    # --- spin type per bowler (arm + direction of turn) ---------------------------------------------------
    sp = d[(d["DeliveryClass"] == "Spin") & d["ok_dev"]]
    info = sp.groupby("Bowler").agg(n=("Deviation", "size"), mean=("Deviation", "mean"),
                                    absmean=("Deviation", lambda s: float(s.abs().mean())), rh=("BowlerRH", "first"))
    def _label(r) -> str:
        if r["n"] < C.SPIN_TYPE_MIN_BALLS:
            return "Unclassified (few balls)"
        ratio = r["mean"] / r["absmean"] if r["absmean"] else 0.0   # +1 = always turns to bowler's right, -1 = always left
        hi, lo = ratio >= C.SPIN_TYPE_MIN_CONSISTENCY, ratio <= -C.SPIN_TYPE_MIN_CONSISTENCY
        if r["rh"]:
            return "Off-spin" if hi else ("Leg-spin" if lo else "Mixed (off/leg)")
        return "Left-arm wrist-spin" if hi else ("Left-arm orthodox" if lo else "Mixed (left-arm)")
    spin_type = info.apply(_label, axis=1) if len(info) else pd.Series(dtype=object)
    d["SpinType"] = d["Bowler"].map(spin_type)
    d.loc[d["DeliveryClass"] != "Spin", "SpinType"] = None

    # --- player tables (used for name resolution) --------------------------------------------------------
    bowlers = d.groupby("Bowler").agg(Balls=("Runs", "size"), Hand=("BowlerHand", "first"),
                                      Class=("DeliveryClass", lambda s: s.mode().iat[0]),
                                      SpinType=("SpinType", "first")).sort_values("Balls", ascending=False)
    batters = d.groupby("Batter").agg(Balls=("Runs", "size"), Hand=("BatterHand", "first")).sort_values("Balls", ascending=False)

    report = build_report(d, df_raw, cfg, unit, parity_mode, bat_conf, bowl_conf, n_medium)
    report["suspect_delivery_type"] = suspects
    report["suspect_class_corrected"] = bool(fix_suspect_class and not parity_mode and suspects)
    report["possible_same_player"] = _possible_same(d)
    if suspects and not report["suspect_class_corrected"]:
        nb = sum(x["balls"] for x in suspects)
        report["notes"].append(f"{len(suspects)} bowlers ({nb} balls) are labelled Seam but bowl at spin speeds (median < 100 kph): " + ", ".join(x["bowler"] for x in suspects) + ". Labels left as supplied.")
    if any(not x["years_overlap"] for x in report["possible_same_player"]):
        report["notes"].append("Possible same player under two names (not merged): " + "; ".join(dict.fromkeys(f"{x['short_name']} / {x['full_name']}" for x in report["possible_same_player"] if not x["years_overlap"])))
    return Prepared(df=d, cfg=cfg, unit=unit, parity_mode=parity_mode, report=report, bowlers=bowlers, batters=batters)


def build_report(d, raw, cfg, unit, parity_mode, bat_conf, bowl_conf, n_medium) -> Dict:
    g = d.groupby(["Match", "Innings", "Over"])
    rows_per_over = g.size()
    last_over = d.groupby(["Match", "Innings"])["Over"].transform("max")
    short = d.groupby(["Match", "Innings", "Over"]).agg(n=("Runs", "size"), last=("Over", "first")).reset_index()
    short["lastover"] = short.set_index(["Match", "Innings"]).index.map(d.groupby(["Match", "Innings"])["Over"].max())
    seam_speed = d.loc[(d["DeliveryClass"] == "Seam") & (d["ReleaseSpeed"] > 0), "ReleaseSpeed"]
    det_unit = None if seam_speed.empty else ("kph" if seam_speed.median() >= 100 else "mph")
    not_ut = ~d["Untracked"]
    def excl(col, lo_hi_col):
        lo, hi = C.RANGES[lo_hi_col]
        return int((not_ut & d[col].notna() & ~d[col].between(lo, hi)).sum())
    rep = {
        "rows": int(len(d)),
        "matches": int(d["Match"].nunique()),
        "years": sorted(int(y) for y in d["Year"].dropna().unique()),
        "tours": int(d["Tour"].nunique()),
        "grounds": int(d["Ground"].nunique()),
        "format": cfg.label,
        "speed_unit_selected": unit,
        "speed_unit_detected": det_unit,
        "parity_mode": parity_mode,
        "untracked_balls": int(d["Untracked"].sum()),
        "untracked_share_pct": round(100 * float(d["Untracked"].mean()), 2),
        "medium_balls_counted_as_seam": 0 if parity_mode else n_medium,
        "other_delivery_types": int((d["DeliveryClass"] == "Other").sum()),
        "overs_with_extra_deliveries": int((rows_per_over > 6).sum()),
        "overs_with_fewer_than_6_rows": int((short["n"] < 6).sum()),
        "short_overs_not_innings_last": int(((short["n"] < 6) & (short["Over"] != short["lastover"])).sum()) if "Over" in short else None,
        "matches_with_one_innings": int((d.groupby("Match")["Innings"].nunique() < 2).sum()),
        "no_contact_balls": int((d["InterceptionX"] <= C.NO_CONTACT).sum()),
        "excluded_per_metric": {
            "BounceX out of range": excl("BounceX", "BounceX"),
            "ReleaseSpeed out of range": excl("ReleaseSpeed", "ReleaseSpeed"),
            "Deviation out of range": excl("Deviation", "Deviation"),
            "Swing out of range": excl("Swing", "Swing"),
            "CreaseY out of range": excl("CreaseY", "CreaseY"),
            "StumpsY out of range": excl("StumpsY", "StumpsY"),
            "ReleaseY out of range": excl("ReleaseY", "ReleaseY"),
        },
        "handedness_conflicts_batters": bat_conf,
        "handedness_conflicts_bowlers": bowl_conf,
        "batter_names": int(d["Batter"].nunique()),
        "bowler_names": int(d["Bowler"].nunique()),
        "name_case_variants_merged": int(raw["BatsmanName"].nunique() - d["Batter"].nunique()) + int(raw["BowlerName"].nunique() - d["Bowler"].nunique()),
    }
    notes: List[str] = []
    if det_unit and det_unit != unit:
        notes.append(f"Speed unit on Home is {unit.upper()} but the data looks like {det_unit.upper()} - check the Speed Unit toggle.")
    if rep["untracked_balls"] and not parity_mode:
        notes.append(f"{rep['untracked_balls']} untracked balls ({rep['untracked_share_pct']}%) keep their runs/balls/wickets but are excluded from every trajectory metric.")
    rep["notes"] = notes
    return rep

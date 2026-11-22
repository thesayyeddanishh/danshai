"""Deterministic analytics for DANsh AI. Python does ALL the maths; the LLM only plans and narrates.

Formulas match the 4 dashboard pages:
  SR = Runs/Balls*100 | batting average = Runs/Dismissals | Economy = Runs/Balls*6
  bowling average = Runs/Wickets | bowling SR = Balls/Wickets | boundary = Runs in {4, 6} | dot = Runs == 0
  Balls = number of rows (wides/no-balls are rows). Average/SR are N/A (None) when there are 0 wickets.
"""
from __future__ import annotations

import difflib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cricket_config import ordered_seam_keys, ordered_spin_keys, seam_speed_ordered_groups, spin_speed_ordered_groups
from . import config as C
from .prep import Prepared, norm_name

DIM_COL = {
    "phase": "Phase", "delivery_class": "DeliveryClass", "bowler_hand": "BowlerHand", "batter_hand": "BatterHand",
    "length": "LengthBin", "line": "LateralZone", "speed_band": "SpeedBand", "ground": "Ground", "year": "Year",
    "innings": "Innings", "over": "Over", "bowling_team": "BowlingTeam", "batting_team": "BattingTeam",
    "over_round": "OverRound", "spin_type": "SpinType", "batter": "Batter", "bowler": "Bowler",
}
BASE_AGG = dict(
    Balls=("Runs", "size"), Runs=("Runs", "sum"), Wkts=("IsWicket", "sum"), Dots=("Dot", "sum"),
    Fours=("Four", "sum"), Sixes=("Six", "sum"), Bdry=("Boundary", "sum"),
    AvgLen=("LenV", "mean"), AvgSpeed=("SpeedV", "mean"), AbsSwing=("AbsSwingV", "mean"), AbsTurn=("AbsDevV", "mean"),
    NetTurn=("DevV", "mean"), NetInSwing=("SwingToLegV", "mean"), Hit=("HitV", "mean"), ContactDist=("ContactDistV", "mean"),
)
BAT_COLS = ["Balls", "Runs", "Dismissals", "SR", "BatAvg", "Dot%", "Bdry%", "Fours", "Sixes", "AvgSpeedFaced"]
BOWL_COLS = ["Balls", "Runs", "Wkts", "Econ", "BowlAvg", "BowlSR", "Dot%", "Bdry%", "Hit%", "AvgLen_m",
             "AvgSpeed", "AvgAbsSwing_deg", "AvgAbsTurn_deg"]
BATTER_METRICS = {  # name -> (column, ascending)
    "runs": ("Runs", False), "balls": ("Balls", False), "strike_rate": ("SR", False), "average": ("BatAvg", False),
    "boundary_pct": ("Bdry%", False), "dot_pct": ("Dot%", True), "dismissals": ("Dismissals", False),
    "fours": ("Fours", False), "sixes": ("Sixes", False),
}
BOWLER_METRICS = {
    "wickets": ("Wkts", False), "economy": ("Econ", True), "bowling_average": ("BowlAvg", True),
    "bowling_strike_rate": ("BowlSR", True), "dot_pct": ("Dot%", False), "boundary_pct": ("Bdry%", True),
    "hit_stumps_pct": ("Hit%", False), "avg_speed": ("AvgSpeed", False), "avg_length": ("AvgLen_m", True),
    "avg_abs_swing": ("AvgAbsSwing_deg", False), "avg_abs_turn": ("AvgAbsTurn_deg", False), "balls": ("Balls", False),
}


# ---------------------------------------------------------------------------------------------
# confidence & intervals
# ---------------------------------------------------------------------------------------------
def wilson(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def bootstrap_mean_ci(x: np.ndarray, scale: float, B: int = 400, seed: int = 7) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if len(x) > 20000:
        x = x[rng.integers(0, len(x), 20000)]
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1) * scale
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def confidence(balls: int, wkts: Optional[int] = None) -> Dict:
    score = 1 - math.exp(-max(balls, 0) / C.CONF_K)
    label = next(lbl for t, lbl in C.CONF_BANDS if score < t)
    warns: List[str] = []
    if balls < 12:
        warns.append(f"VERY LOW SAMPLE: only {balls} balls - indicative only.")
    elif balls < 30:
        warns.append(f"Low sample: {balls} balls.")
    if wkts is not None and wkts < C.MIN_WICKETS_FOR_AVG:
        warns.append(f"Average / strike rate rest on only {wkts} wicket(s).")
    return {"balls": int(balls), "score_pct": round(100 * score), "label": label, "warnings": warns}


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------
def _aggregate(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    if not by:
        t = df.assign(_all="All").groupby("_all").agg(**BASE_AGG)
        t.index.name = "Group"
        return t
    return df.groupby(by, dropna=True, sort=False).agg(**BASE_AGG)


def _with_rates(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    b = t["Balls"].replace(0, np.nan)
    w = t["Wkts"].replace(0, np.nan)
    t["Dismissals"] = t["Wkts"]
    t["SR"] = t["Runs"] / b * 100
    t["BatAvg"] = t["Runs"] / w
    t["Econ"] = t["Runs"] / b * 6
    t["BowlAvg"] = t["Runs"] / w
    t["BowlSR"] = t["Balls"] / w
    t["Dot%"] = t["Dots"] / b * 100
    t["Bdry%"] = t["Bdry"] / b * 100
    t["Hit%"] = t["Hit"] * 100
    t["AvgLen_m"] = t["AvgLen"]
    t["AvgSpeedFaced"] = t["AvgSpeed"]
    t["AvgAbsSwing_deg"] = t["AbsSwing"]
    t["AvgAbsTurn_deg"] = t["AbsTurn"]
    t["Confidence"] = [confidence(int(n))["label"] for n in t["Balls"]]
    return t


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def to_records(t: pd.DataFrame, cols: List[str], index_name: Optional[str] = None, max_rows: int = 40) -> List[Dict]:
    out = []
    for idx, row in t.head(max_rows).iterrows():
        rec: Dict = {}
        if isinstance(idx, tuple):
            for n, v in zip(t.index.names, idx):
                rec[n] = _clean(v)
        else:
            rec[index_name or (t.index.name or "Group")] = _clean(idx)
        for c in cols + ["Confidence"]:
            if c in t.columns:
                rec[c] = _clean(row[c])
        out.append(rec)
    return out


def _cols(role: str) -> List[str]:
    return BAT_COLS if role == "batter" else BOWL_COLS


def _order_values(dim: str, values: List, prep: Prepared, df: pd.DataFrame) -> List:
    cfg = prep.cfg
    if dim == "phase":
        pref = ["Powerplay (1-6)", "Middle (7-16)", "Death (17-20)", "Powerplay (1-10)", "Middle (11-40)", "Death (41-50)", "All"]
        return [p for p in pref if p in values]
    if dim == "length":
        keys = ordered_seam_keys(cfg) + [k for k in ordered_spin_keys(cfg) if k not in ordered_seam_keys(cfg)]
        return [k for k in keys if k in values] + [v for v in values if v not in keys]
    if dim == "line":
        base = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
        return [k for k in base if k in values]
    if dim == "speed_band":
        groups = seam_speed_ordered_groups(cfg, prep.unit)[::-1] + spin_speed_ordered_groups(cfg, prep.unit)[::-1]
        seen: List = []
        for g in groups:
            if g in values and g not in seen:
                seen.append(g)
        return seen + [v for v in values if v not in seen]
    if dim in ("bowler_hand", "batter_hand"):
        return [v for v in ["Right", "Left"] if v in values]
    if dim in ("year", "innings", "over"):
        return sorted(values)
    return values


def group_table(df: pd.DataFrame, prep: Prepared, role: str, by: List[str]) -> Tuple[pd.DataFrame, int]:
    """Group `df` by the requested dims. Returns (table, rows_without_value_for_the_split)."""
    cols = [DIM_COL[b] for b in by]
    valid = df.dropna(subset=cols) if cols else df
    t = _with_rates(_aggregate(valid, cols))
    if cols:
        t.index.names = by
        if len(by) == 1:
            order = _order_values(by[0], list(t.index), prep, valid)
            t = t.reindex(order)
        else:
            t = t.sort_values("Balls", ascending=False)
        if by[0] in ("batter", "bowler"):
            t = t.sort_values("Balls", ascending=False)
    return t, int(len(df) - len(valid))


def summary_row(df: pd.DataFrame, role: str, with_ci: bool = True) -> Dict:
    """One-row headline summary (+ confidence intervals)."""
    if df.empty:
        return {"Balls": 0}
    t = _with_rates(_aggregate(df, []))
    row = t.iloc[0]
    rec = {c: _clean(row[c]) for c in _cols(role)}
    n, w = int(row["Balls"]), int(row["Wkts"])
    if with_ci:
        lo, hi = wilson(float(row["Bdry"]), n)
        rec["Bdry%_95CI"] = [_clean(lo), _clean(hi)]
        lo, hi = wilson(float(row["Dots"]), n)
        rec["Dot%_95CI"] = [_clean(lo), _clean(hi)]
        key = "SR" if role == "batter" else "Econ"
        lo, hi = bootstrap_mean_ci(df["Runs"].to_numpy(), 100.0 if role == "batter" else 6.0)
        rec[f"{key}_95CI"] = [_clean(lo), _clean(hi)]
    if role == "bowler":
        rec["NetTurn_deg(+ = bowler's right)"] = _clean(row["NetTurn"])
        rec["NetInSwing_deg(+ = toward batter's leg)"] = _clean(row["NetInSwing"])
    rec["confidence"] = confidence(n, w)
    return rec


# ---------------------------------------------------------------------------------------------
# name resolution & filters
# ---------------------------------------------------------------------------------------------
class NeedsClarification(Exception):
    def __init__(self, message: str, candidates: List[str]):
        super().__init__(message)
        self.message, self.candidates = message, candidates


def resolve_player(prep: Prepared, query: str, role: Optional[str] = None) -> str:
    q = norm_name(query)
    pools = []
    if role in (None, "batter"):
        pools.append(prep.batters["Balls"])
    if role in (None, "bowler"):
        pools.append(prep.bowlers["Balls"])
    balls = pd.concat(pools).groupby(level=0).sum().sort_values(ascending=False)
    names = list(balls.index)
    if q in balls.index:
        return q
    toks = q.split()
    cand = [n for n in names if all(any(w.startswith(t) for w in n.split()) for t in toks)]
    if not cand:
        cand = [n for n in names if all(t in n for t in toks)]
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        raise NeedsClarification(f"'{query}' matches several players - which one?", cand[:8])
    close = difflib.get_close_matches(q, names, n=5, cutoff=0.6)
    raise NeedsClarification(f"No player matching '{query}' in this data." + (" Did you mean one of these?" if close else ""), close)


def _match_values(series: pd.Series, queries: List[str], what: str) -> List[str]:
    vals = sorted(series.dropna().unique())
    out: List[str] = []
    for q in queries:
        qn = norm_name(q).replace(" ", "-")
        hit = [v for v in vals if qn in v.replace(" ", "-") or norm_name(q) in v]
        if not hit:
            raise NeedsClarification(f"No {what} matching '{q}'.", difflib.get_close_matches(norm_name(q), vals, n=5, cutoff=0.4))
        out += hit
    return sorted(set(out))


HAND = {"right": "Right", "r": "Right", "rh": "Right", "rhb": "Right", "left": "Left", "l": "Left", "lh": "Left", "lhb": "Left"}


def apply_filters(prep: Prepared, f: Optional[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    d, f, applied = prep.df, (f or {}), []
    if f.get("years"):
        ys = [int(y) for y in f["years"]]
        d = d[d["Year"].isin(ys)]; applied.append(f"years={ys}")
    if f.get("tours"):
        v = _match_values(d["Tour"], f["tours"], "tour"); d = d[d["Tour"].isin(v)]; applied.append(f"tours={v}")
    if f.get("grounds"):
        v = _match_values(d["Ground"], f["grounds"], "ground"); d = d[d["Ground"].isin(v)]; applied.append(f"grounds={v}")
    if f.get("phase") and f["phase"] != "All":
        opts = [p for p in d["Phase"].unique()]
        hit = [p for p in opts if f["phase"].lower() in p.lower()]
        if not hit:
            raise NeedsClarification(f"Phase '{f['phase']}' not available for {prep.cfg.label}.", sorted(opts))
        d = d[d["Phase"].isin(hit)]; applied.append(f"phase={hit}")
    if f.get("over_from") is not None or f.get("over_to") is not None:
        lo, hi = int(f.get("over_from") or 1), int(f.get("over_to") or 999)
        d = d[(d["Over"] >= lo) & (d["Over"] <= hi)]; applied.append(f"overs {lo}-{hi}")
    if f.get("innings"):
        d = d[d["Innings"] == int(f["innings"])]; applied.append(f"innings={int(f['innings'])}")
    if f.get("delivery_class"):
        dc = str(f["delivery_class"]).title(); dc = "Seam" if dc in ("Pace", "Fast", "Quick", "Medium") else dc
        d = d[d["DeliveryClass"] == dc]; applied.append(f"delivery_class={dc}")
    for key, col in (("bowler_hand", "BowlerHand"), ("batter_hand", "BatterHand")):
        if f.get(key):
            h = HAND.get(str(f[key]).strip().lower())
            if not h:
                raise NeedsClarification(f"{key} must be Right or Left.", ["Right", "Left"])
            d = d[d[col] == h]; applied.append(f"{key}={h}")
    for key, col, role in (("batters", "Batter", "batter"), ("bowlers", "Bowler", "bowler")):
        if f.get(key):
            names = [resolve_player(prep, n, role) for n in f[key]]
            d = d[d[col].isin(names)]; applied.append(f"{key}={names}")
    for key, col, what in (("batting_team", "BattingTeam", "team"), ("bowling_team", "BowlingTeam", "team")):
        if f.get(key):
            v = _match_values(d[col], f[key], what); d = d[d[col].isin(v)]; applied.append(f"{key}={v}")
    if f.get("over_round"):
        want = "Over the wicket" if "over" in str(f["over_round"]).lower() else "Round the wicket"
        d = d[d["OverRound"] == want]; applied.append(f"over_round={want}")
    if f.get("spin_type"):
        d = d[d["SpinType"].fillna("").str.lower().str.contains(str(f["spin_type"]).lower())]; applied.append(f"spin_type~{f['spin_type']}")
    if f.get("length"):
        want = [str(x).strip().lower() for x in f["length"]]
        d = d[d["LengthBin"].fillna("").str.lower().isin(want)]; applied.append(f"length={f['length']}")
    if f.get("line"):
        want = [str(x).strip().upper() for x in f["line"]]
        d = d[d["LateralZone"].isin(want)]; applied.append(f"line={want}")
    return d, applied


# ---------------------------------------------------------------------------------------------
# high-level analyses used by the tools
# ---------------------------------------------------------------------------------------------
def _entity_df(prep: Prepared, df: pd.DataFrame, entity: str, role: str, merge_with: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    """`merge_with` combines extra name variants - only when the user explicitly confirmed they are the same player."""
    names = [resolve_player(prep, entity, role)] + [resolve_player(prep, x, role) for x in (merge_with or [])]
    names = list(dict.fromkeys(names))
    return df[df["Batter" if role == "batter" else "Bowler"].isin(names)], " + ".join(names)


def same_player_hints(prep: Prepared, name: str) -> List[Dict]:
    return [x for x in prep.report.get("possible_same_player", []) if name in (x["short_name"], x["full_name"])]


DEFAULT_BREAKDOWNS = {
    "batter": ["delivery_class", "spin_type", "bowler_hand", "phase", "length", "line"],
    "bowler": ["batter_hand", "phase", "length", "line", "speed_band", "over_round"],
}


def profile(prep: Prepared, entity: str, role: str, filters: Optional[Dict], breakdowns: Optional[List[str]], merge_with: Optional[List[str]] = None) -> Dict:
    base, applied = apply_filters(prep, filters)
    d, name = _entity_df(prep, base, entity, role, merge_with)
    first = name.split(" + ")[0]
    hints = same_player_hints(prep, first) if not merge_with else []
    out: Dict = {"entity": name, "role": role, "filters_applied": applied, "format": prep.cfg.label}
    if hints:
        out["possible_same_player_NOT_merged"] = hints
    if d.empty:
        out["note"] = "No deliveries for this player with these filters."
        out["confidence"] = confidence(0)
        return out
    meta = prep.bowlers.loc[first] if role == "bowler" and first in prep.bowlers.index else None
    if role == "bowler" and meta is not None:
        out["bowler_info"] = {"arm": meta["Hand"], "type": meta["Class"], "spin_type": meta["SpinType"] if isinstance(meta["SpinType"], str) else None}
    if role == "batter" and first in prep.batters.index:
        out["batter_info"] = {"hand": prep.batters.loc[first, "Hand"]}
    out["summary"] = summary_row(d, role)
    out["confidence"] = out["summary"]["confidence"]
    tables: Dict = {}
    notes: List[str] = []
    for b in (breakdowns or DEFAULT_BREAKDOWNS[role]):
        if b not in DIM_COL:
            continue
        t, dropped = group_table(d, prep, role, [b])
        if t.empty:
            continue
        tables[f"by_{b}"] = to_records(t, _cols(role), index_name=b)
        if dropped and b in ("length", "line", "speed_band", "over_round", "spin_type"):
            notes.append(f"by_{b}: {dropped} balls have no valid value for this split and are left out.")
    out["tables"] = tables
    out["notes"] = notes
    return out


def leaderboard(prep: Prepared, role: str, metric: str, filters: Optional[Dict], min_balls: int, top_n: int, order: Optional[str]) -> Dict:
    metrics = BATTER_METRICS if role == "batter" else BOWLER_METRICS
    if metric not in metrics:
        raise NeedsClarification(f"Unknown metric '{metric}' for {role}.", sorted(metrics))
    col, asc = metrics[metric]
    if order in ("asc", "desc"):
        asc = order == "asc"
    d, applied = apply_filters(prep, filters)
    t, _ = group_table(d, prep, role, ["batter" if role == "batter" else "bowler"])
    t = t[t["Balls"] >= min_balls].dropna(subset=[col])
    t = t.sort_values([col, "Balls"], ascending=[asc, False]).head(top_n)
    cols = _cols(role)
    out = {"role": role, "metric": metric, "ordered": "ascending" if asc else "descending", "min_balls": min_balls,
           "filters_applied": applied, "format": prep.cfg.label, "rows": to_records(t, cols, index_name="Player")}
    out["confidence"] = confidence(int(t["Balls"].min()) if len(t) else 0)
    out["notes"] = ["Rows show a Confidence label from balls faced/bowled; low-sample rows are kept, not hidden."]
    return out


def compare(prep: Prepared, entities: List[str], role: str, filters: Optional[Dict], by: Optional[str]) -> Dict:
    base, applied = apply_filters(prep, filters)
    rows, names, dfs = [], [], []
    for e in entities:
        d, n = _entity_df(prep, base, e, role)
        names.append(n); dfs.append(d)
        rec = {"Player": n}; rec.update(summary_row(d, role, with_ci=False) if not d.empty else {"Balls": 0})
        rec.pop("confidence", None)
        rec["Confidence"] = confidence(len(d), int(d["IsWicket"].sum()))["label"]
        rows.append(rec)
    out: Dict = {"role": role, "players": names, "filters_applied": applied, "format": prep.cfg.label, "side_by_side": rows}
    if by in DIM_COL:
        col = DIM_COL[by]
        both = pd.concat(dfs)
        t, _ = group_table(both, prep, role, ["batter" if role == "batter" else "bowler", by])
        out[f"by_{by}"] = to_records(t, _cols(role), max_rows=60)
    out["confidence"] = confidence(min(len(x) for x in dfs) if dfs else 0)
    return out


def matchup(prep: Prepared, bowler: str, batter: str, filters: Optional[Dict]) -> Dict:
    base, applied = apply_filters(prep, filters)
    bw, bt = resolve_player(prep, bowler, "bowler"), resolve_player(prep, batter, "batter")
    d = base[(base["Bowler"] == bw) & (base["Batter"] == bt)]
    out: Dict = {"bowler": bw, "batter": bt, "filters_applied": applied, "format": prep.cfg.label}
    if d.empty:
        out["balls"] = 0
        out["note"] = "These two have not met in the filtered data."
        out["confidence"] = confidence(0)
        return out
    s = summary_row(d, "bowler")
    out["head_to_head"] = {**s, "Batter_SR": _clean(float(d["Runs"].sum() / len(d) * 100)),
                           "Batter_dismissals": int(d["IsWicket"].sum())}
    out["confidence"] = s["confidence"]
    for b in ("length", "line"):
        t, _ = group_table(d, prep, "bowler", [b])
        if not t.empty:
            out[f"by_{b}"] = to_records(t, ["Balls", "Runs", "Wkts", "Econ", "Dot%", "Bdry%"], index_name=b)
    cls = d["DeliveryClass"].mode().iat[0]
    bhand = prep.bowlers.loc[bw, "Hand"]
    ctx_bat = base[(base["Batter"] == bt) & (base["DeliveryClass"] == cls) & (base["BowlerHand"] == bhand)]
    ctx_bowl = base[(base["Bowler"] == bw) & (base["BatterHand"] == prep.batters.loc[bt, "Hand"])]
    out["context"] = {
        f"{bt} vs all {bhand.lower()}-arm {cls.lower()} bowlers": summary_row(ctx_bat, "batter", with_ci=False) if not ctx_bat.empty else None,
        f"{bw} vs all {prep.batters.loc[bt, 'Hand'].lower()}-handers": summary_row(ctx_bowl, "bowler", with_ci=False) if not ctx_bowl.empty else None,
    }
    for v in out["context"].values():
        if isinstance(v, dict):
            v.pop("confidence", None)
    return out


def ground_report(prep: Prepared, ground: str, filters: Optional[Dict]) -> Dict:
    f = dict(filters or {}); f["grounds"] = [ground]
    d, applied = apply_filters(prep, f)
    out: Dict = {"filters_applied": applied, "format": prep.cfg.label}
    if d.empty:
        out["note"] = "No deliveries at this ground with these filters."
        out["confidence"] = confidence(0)
        return out
    n_inn = d.groupby(["Match", "Innings"]).ngroups
    s = summary_row(d, "bowler")
    out["overall"] = {"matches": int(d["Match"].nunique()), "innings": int(n_inn), "balls": len(d),
                      "runs_per_over(bat runs only)": _clean(d["Runs"].sum() / len(d) * 6),
                      "bat_runs_per_innings(extras not in data)": _clean(d["Runs"].sum() / n_inn),
                      "wickets_per_innings": _clean(d["IsWicket"].sum() / n_inn),
                      "Bdry%": s["Bdry%"], "Dot%": s["Dot%"], "AvgSpeed_all": s["AvgSpeed"]}
    out["confidence"] = confidence(len(d))
    for b in ("delivery_class", "phase", "innings", "batter_hand"):
        t, _ = group_table(d, prep, "bowler", [b])
        if not t.empty:
            out[f"by_{b}"] = to_records(t, ["Balls", "Runs", "Wkts", "Econ", "BowlAvg", "BowlSR", "Dot%", "Bdry%", "AvgSpeed", "AvgLen_m", "AvgAbsSwing_deg", "AvgAbsTurn_deg"], index_name=b)
    out["note"] = "Runs are bat runs only (the data has no extras column)."
    return out


def overview(prep: Prepared) -> Dict:
    d = prep.df
    return {
        "format": prep.cfg.label, "rows": len(d), "matches": int(d["Match"].nunique()),
        "years": sorted(int(y) for y in d["Year"].dropna().unique()),
        "grounds": d["Ground"].value_counts().head(15).to_dict(),
        "teams": sorted(d["BattingTeam"].unique()),
        "top_batters_by_balls": prep.batters.head(10)["Balls"].to_dict(),
        "top_bowlers_by_balls": prep.bowlers.head(10)["Balls"].to_dict(),
        "data_quality": {k: v for k, v in prep.report.items() if k not in ("handedness_conflicts_bowlers",)},
    }

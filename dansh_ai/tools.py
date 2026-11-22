"""The fixed set of functions the LLM may call (no code generation). Every number comes from metrics.py."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import charts as CH
from . import config as C
from . import metrics as M
from .prep import Prepared

_STR_LIST = {"type": "array", "items": {"type": "string"}}
FILTERS = {
    "type": "object",
    "description": "Optional filters. Only use values the user asked for.",
    "properties": {
        "years": {"type": "array", "items": {"type": "integer"}},
        "tours": _STR_LIST, "grounds": {**_STR_LIST, "description": "ground name fragments, e.g. ['premadasa']"},
        "phase": {"type": "string", "enum": ["All", "Powerplay", "Middle", "Death"]},
        "over_from": {"type": "integer"}, "over_to": {"type": "integer"},
        "innings": {"type": "integer", "enum": [1, 2, 3, 4]},
        "delivery_class": {"type": "string", "enum": ["Seam", "Spin"], "description": "quick/pace/fast/medium = Seam"},
        "bowler_hand": {"type": "string", "enum": ["Right", "Left"]}, "batter_hand": {"type": "string", "enum": ["Right", "Left"]},
        "batters": _STR_LIST, "bowlers": _STR_LIST, "batting_team": _STR_LIST, "bowling_team": _STR_LIST,
        "over_round": {"type": "string", "enum": ["Over the wicket", "Round the wicket"]},
        "spin_type": {"type": "string", "description": "e.g. leg-spin, off-spin, orthodox, wrist"},
        "length": _STR_LIST, "line": {"type": "array", "items": {"type": "string", "enum": ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]}},
    },
}
ROLE = {"type": "string", "enum": ["batter", "bowler"]}
CHARTS = {"type": "array", "items": {"type": "string", "enum": ["pitch_map", "beehive", "wagon_wheel", "length_bars", "speed_hist"]}}
DIMS = {"type": "string", "enum": list(C.GROUP_DIMS)}


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


TOOLS = [
    _fn("dataset_overview", "Summary of the loaded data: format, years, grounds, teams, top players, data-quality notes.", {}, []),
    _fn("search_players", "Find player names in the data (use when a name is unclear).",
        {"query": {"type": "string"}, "role": ROLE}, ["query"]),
    _fn("player_profile", "Full profile of ONE batter or bowler with breakdown tables and optional charts. Use for strengths/weaknesses/overview questions.",
        {"player": {"type": "string"}, "role": ROLE, "filters": FILTERS,
         "breakdowns": {"type": "array", "items": DIMS, "description": "splits to include; omit for sensible defaults"},
         "merge_with": {**_STR_LIST, "description": "extra name variants of the SAME player - only if the user confirmed"},
         "charts": CHARTS}, ["player", "role"]),
    _fn("group_stats", "Stats split by one or two dimensions across all (filtered) data, e.g. economy by phase for spinners.",
        {"role": ROLE, "group_by": {"type": "array", "items": DIMS, "minItems": 1, "maxItems": 2}, "filters": FILTERS,
         "chart_metric": {"type": "string", "description": "optional metric column to chart, e.g. Econ, SR, Dot%, Bdry%, Hit%"}}, ["role", "group_by"]),
    _fn("leaderboard", "Rank batters or bowlers by a metric.",
        {"role": ROLE, "metric": {"type": "string", "enum": sorted(set(M.BATTER_METRICS) | set(M.BOWLER_METRICS))},
         "filters": FILTERS, "min_balls": {"type": "integer"}, "top_n": {"type": "integer"}, "order": {"type": "string", "enum": ["asc", "desc"]}},
        ["role", "metric"]),
    _fn("compare_players", "Side-by-side comparison of 2-4 players of the same role.",
        {"players": {**_STR_LIST, "minItems": 2, "maxItems": 4}, "role": ROLE, "filters": FILTERS, "by": DIMS}, ["players", "role"]),
    _fn("matchup", "Bowler vs batter head-to-head with length/line splits and context.",
        {"bowler": {"type": "string"}, "batter": {"type": "string"}, "filters": FILTERS, "charts": CHARTS}, ["bowler", "batter"]),
    _fn("ground_report", "Ground statistics: scoring, pace vs spin, phases, innings 1 vs 2.",
        {"ground": {"type": "string"}, "filters": FILTERS}, ["ground"]),
    _fn("make_chart", "Draw one chart for a player or for all filtered data.",
        {"chart": {"type": "string", "enum": ["pitch_map", "beehive", "wagon_wheel", "length_bars", "speed_hist", "bar", "trend"]},
         "role": ROLE, "player": {"type": "string"}, "filters": FILTERS, "group_by": DIMS,
         "metric": {"type": "string", "description": "for bar/trend: Econ, SR, Dot%, Bdry%, Hit%, BowlAvg, BatAvg, AvgSpeed"}},
        ["chart", "role"]),
]

METRIC_COLS = {"econ": "Econ", "economy": "Econ", "sr": "SR", "strike_rate": "SR", "dot%": "Dot%", "dot_pct": "Dot%",
               "bdry%": "Bdry%", "boundary_pct": "Bdry%", "hit%": "Hit%", "bowlavg": "BowlAvg", "batavg": "BatAvg",
               "bowlsr": "BowlSR", "avgspeed": "AvgSpeed", "avglen": "AvgLen_m", "wkts": "Wkts", "runs": "Runs"}


@dataclass
class ToolOutput:
    result: Dict[str, Any]
    figures: List[Tuple[str, Any]] = field(default_factory=list)
    tables: List[Tuple[str, pd.DataFrame]] = field(default_factory=list)


def _tables_from(result: Dict) -> List[Tuple[str, pd.DataFrame]]:
    out = []
    if isinstance(result.get("summary"), dict):
        flat = {k: v for k, v in result["summary"].items() if not isinstance(v, (dict, list))}
        out.append(("Summary", pd.DataFrame([flat])))
    for k, v in result.items():
        if (k.startswith("by_") or k in ("rows", "side_by_side")) and isinstance(v, list) and v:
            out.append((k.replace("_", " ").title(), pd.DataFrame(v)))
    for k, v in (result.get("tables") or {}).items():
        if isinstance(v, list) and v:
            out.append((k.replace("_", " ").title(), pd.DataFrame(v)))
    return out


def _title(entity: str, applied: List[str]) -> str:
    return entity + (f"  [{'; '.join(applied)[:70]}]" if applied else "")


def _figs(names: List[str], d: pd.DataFrame, prep: Prepared, role: str, title: str) -> List[Tuple[str, Any]]:
    out = []
    for n in names or []:
        if n == "pitch_map":
            out.append((f"Pitch map - {title}", CH.pitch_map(d, prep, title)))
        elif n == "beehive":
            out.append((f"Beehive - {title}", CH.beehive(d, prep, title)))
        elif n == "wagon_wheel":
            out.append((f"Wagon wheel - {title}", CH.wagon_wheel(d, prep, title)))
        elif n == "length_bars":
            out.append((f"Length - {title}", CH.length_bars(d, prep, role, title)))
        elif n == "speed_hist":
            out.append((f"Speed - {title}", CH.speed_hist(d, prep, title)))
    return out


def execute(name: str, args: Dict[str, Any], prep: Prepared, state: Dict) -> ToolOutput:
    try:
        return _execute(name, args or {}, prep, state)
    except M.NeedsClarification as e:
        return ToolOutput({"needs_clarification": e.message, "candidates": e.candidates})
    except Exception as e:  # never crash the chat; tell the model what went wrong
        return ToolOutput({"error": f"{type(e).__name__}: {e}"})


def _execute(name: str, a: Dict, prep: Prepared, state: Dict) -> ToolOutput:
    f = a.get("filters")
    if name == "dataset_overview":
        return ToolOutput(M.overview(prep))
    if name == "search_players":
        q, role = a["query"], a.get("role")
        try:
            return ToolOutput({"match": M.resolve_player(prep, q, role)})
        except M.NeedsClarification as e:
            return ToolOutput({"message": e.message, "candidates": e.candidates})
    if name == "player_profile":
        role = a["role"]
        r = M.profile(prep, a["player"], role, f, a.get("breakdowns"), a.get("merge_with"))
        figs: List = []
        if "summary" in r:
            base, applied = M.apply_filters(prep, f)
            d, nm = M._entity_df(prep, base, a["player"], role, a.get("merge_with"))
            figs = _figs(a.get("charts"), d, prep, role, _title(nm, applied))
            state.update({"last_entity": nm, "last_role": role, "last_filters": f or {}})
        return ToolOutput(r, figs, _tables_from(r))
    if name == "group_stats":
        role, by = a["role"], a["group_by"]
        d, applied = M.apply_filters(prep, f)
        t, dropped = M.group_table(d, prep, role, by)
        r = {"role": role, "group_by": by, "filters_applied": applied, "format": prep.cfg.label,
             "rows_without_value_for_split": dropped, "by_" + "_".join(by): M.to_records(t, M._cols(role), max_rows=60),
             "confidence": M.confidence(int(t["Balls"].min()) if len(t) else 0)}
        figs = []
        col = METRIC_COLS.get(str(a.get("chart_metric", "")).lower().replace(" ", ""), a.get("chart_metric"))
        if col in t.columns:
            figs.append((f"{col} by {' / '.join(by)}", CH.bar_chart(t, col, f"{col} by {' / '.join(by)}")))
        state.update({"last_role": role, "last_filters": f or {}})
        return ToolOutput(r, figs, _tables_from(r))
    if name == "leaderboard":
        r = M.leaderboard(prep, a["role"], a["metric"], f, int(a.get("min_balls") or C.MIN_BALLS_LEADERBOARD),
                          int(a.get("top_n") or 10), a.get("order"))
        state.update({"last_role": a["role"], "last_filters": f or {}})
        return ToolOutput(r, [], _tables_from(r))
    if name == "compare_players":
        r = M.compare(prep, a["players"], a["role"], f, a.get("by"))
        state.update({"last_role": a["role"], "last_filters": f or {}, "last_entity": " vs ".join(r["players"])})
        return ToolOutput(r, [], _tables_from(r))
    if name == "matchup":
        r = M.matchup(prep, a["bowler"], a["batter"], f)
        figs = []
        if r.get("head_to_head") and a.get("charts"):
            base, applied = M.apply_filters(prep, f)
            d = base[(base["Bowler"] == r["bowler"]) & (base["Batter"] == r["batter"])]
            figs = _figs(a["charts"], d, prep, "bowler", f"{r['bowler']} to {r['batter']}")
        state.update({"last_entity": f"{r['bowler']} vs {r['batter']}", "last_filters": f or {}})
        tables = _tables_from({k: v for k, v in r.items() if k != "head_to_head"})
        if isinstance(r.get("head_to_head"), dict):
            flat = {k: v for k, v in r["head_to_head"].items() if not isinstance(v, (dict, list))}
            tables.insert(0, ("Head to head", pd.DataFrame([flat])))
        return ToolOutput(r, figs, tables)
    if name == "ground_report":
        r = M.ground_report(prep, a["ground"], f)
        state.update({"last_entity": a["ground"], "last_filters": f or {}})
        return ToolOutput(r, [], _tables_from(r))
    if name == "make_chart":
        role, chart = a["role"], a["chart"]
        base, applied = M.apply_filters(prep, f)
        label = "all deliveries"
        d = base
        if a.get("player"):
            d, label = M._entity_df(prep, base, a["player"], role)
            state.update({"last_entity": label, "last_role": role})
        if d.empty:
            return ToolOutput({"note": "No deliveries match these filters.", "confidence": M.confidence(0)})
        title = _title(label, applied)
        if chart in ("bar", "trend"):
            by = a.get("group_by") or ("year" if chart == "trend" else "phase")
            t, _ = M.group_table(d, prep, role, [by])
            col = METRIC_COLS.get(str(a.get("metric", "")).lower().replace(" ", ""), a.get("metric")) or ("SR" if role == "batter" else "Econ")
            if col not in t.columns:
                return ToolOutput({"error": f"Unknown metric '{col}'", "allowed": sorted(set(METRIC_COLS.values()))})
            fig = CH.trend(t, col, f"{col} by {by} - {title}") if chart == "trend" else CH.bar_chart(t, col, f"{col} by {by} - {title}")
            r = {"chart": chart, "metric": col, "group_by": by, "filters_applied": applied, "by_" + by: M.to_records(t, M._cols(role)),
                 "confidence": M.confidence(len(d))}
            return ToolOutput(r, [(f"{col} by {by}", fig)], _tables_from(r))
        figs = _figs([chart], d, prep, role, title)
        return ToolOutput({"chart": chart, "balls_in_chart": len(d), "filters_applied": applied, "confidence": M.confidence(len(d))}, figs)
    return ToolOutput({"error": f"Unknown tool '{name}'"})


def shrink_json(result: Dict, limit: int = C.TOOL_RESULT_MAX_CHARS) -> str:
    """Serialise a tool result for the model, trimming long tables if needed."""
    s = json.dumps(result, default=str, ensure_ascii=False)
    for rows in (25, 12, 6):
        if len(s) <= limit:
            break
        trimmed = {k: (v[:rows] if isinstance(v, list) and v and isinstance(v[0], dict) else v) for k, v in result.items()}
        s = json.dumps(trimmed, default=str, ensure_ascii=False)
    return s[:limit]

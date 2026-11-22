"""DANsh AI tests. Run:  python tests/test_dansh_ai.py   (or pytest). No network needed.

Set DANSH_TEST_DATA to a CSV/XLSX with the 32 DANshboard columns (default: /mnt/user-data/uploads/LPL.xlsx).
`parity_mode=True` switches cleaning off, so the bot must equal the 4 existing pages' formulas exactly.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cricket_config import apply_match_phase_filter, hitting_stumps_mask, pitch_bin_percentages, get_pitch_bins, resolve_format  # noqa: E402
from dansh_ai import metrics as M  # noqa: E402
from dansh_ai import tools as T  # noqa: E402
from dansh_ai.llm import GroqClient, GroqError, run_turn, unverified_numbers  # noqa: E402
from dansh_ai.prep import lateral_zone, prepare, wagon_area  # noqa: E402

DATA = os.environ.get("DANSH_TEST_DATA", "/mnt/user-data/uploads/LPL.xlsx")
_cache = {}


def raw() -> pd.DataFrame:
    if "raw" not in _cache:
        _cache["raw"] = pd.read_excel(DATA) if DATA.lower().endswith("xlsx") else pd.read_csv(DATA)
    return _cache["raw"]


def prep(parity=False, **kw):
    key = ("p", parity, tuple(sorted(kw.items())))
    if key not in _cache:
        _cache[key] = prepare(raw(), "men_t20i", "kph", parity_mode=parity, **kw)
    return _cache[key]


def approx(a, b, tol=0.006):
    assert a is not None and abs(a - b) <= tol, f"{a} != {b}"


# ---------------------------------------------------------------------------------------------
# PARITY with the existing pages (page logic re-implemented independently from the page source)
# ---------------------------------------------------------------------------------------------
def page_wagon(row):  # verbatim logic of calculate_scoring_wagon
    LX, LY, RH = row["LandingX"], row["LandingY"], row["IsBatsmanRightHanded"]
    if RH is None or LX is None or LY is None or row["Runs"] == 0:
        return None
    at = lambda n, d: np.arctan(n / d) if d != 0 else np.nan
    if RH == True:
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        elif LX > 0 and LY < 0:
            if at(LY, LX) < np.pi / -4: return "COVER"
            elif at(LX, LY) <= np.pi / -4: return "LONG OFF"
        elif LX > 0 and LY >= 0:
            if at(LY, LX) >= np.pi / 4: return "SQUARE LEG"
            elif at(LY, LX) <= np.pi / 4: return "LONG ON"
    else:
        if LX <= 0 and LY > 0: return "THIRD MAN"
        elif LX <= 0 and LY <= 0: return "FINE LEG"
        elif LX > 0 and LY < 0:
            if at(LY, LX) < np.pi / -4: return "SQUARE LEG"
            elif at(LX, LY) <= np.pi / -4: return "LONG ON"
        elif LX > 0 and LY >= 0:
            if at(LY, LX) >= np.pi / 4: return "COVER"
            elif at(LY, LX) <= np.pi / 4: return "LONG OFF"
    return None


def page_zone(y, z, rh):  # verbatim logic of assign_lateral_zone
    if z > 1.85: return None
    if rh == True:
        if 0.115 < y <= 1.75: return "LEG"
        elif -0.115 < y <= 0.115: return "STUMPS"
        elif -0.415 < y <= -0.115: return "OUTSIDE OFF"
        elif -1.245 < y <= -0.415: return "WAY OUTSIDE OFF"
    else:
        if -1.75 <= y < -0.115: return "LEG"
        elif -0.115 <= y < 0.115: return "STUMPS"
        elif 0.115 <= y < 0.415: return "OUTSIDE OFF"
        elif 0.415 <= y < 1.245: return "WAY OUTSIDE OFF"
    return None


def test_wagon_geometry_matches_page():
    r = raw()
    mine = wagon_area(r["LandingX"], r["LandingY"], r["IsBatsmanRightHanded"].astype(bool), r["Runs"])
    page = r.apply(page_wagon, axis=1).to_numpy(dtype=object)
    n = lambda x: x if isinstance(x, str) else None   # pandas turns None into NaN
    bad = sum(1 for a, b in zip(mine, page) if n(a) != n(b))
    assert bad == 0, f"{bad} wagon-wheel sector mismatches"


def test_lateral_zone_matches_page():
    r = raw()
    mine = lateral_zone(r["CreaseY"], r["CreaseZ"], r["IsBatsmanRightHanded"].astype(bool))
    page = [page_zone(y, z, h) for y, z, h in zip(r["CreaseY"], r["CreaseZ"], r["IsBatsmanRightHanded"])]
    n = lambda x: x if isinstance(x, str) else None
    bad = sum(1 for a, b in zip(mine, page) if n(a) != n(b))
    assert bad == 0, f"{bad} lateral-zone mismatches"


def test_phase_matches_config():
    p = prep(parity=True)
    cfg = resolve_format("men_t20i")
    for ph in ("Powerplay (1-6)", "Middle (7-16)", "Death (17-20)"):
        assert (p.df["Phase"] == ph).sum() == len(apply_match_phase_filter(p.df, ph, cfg))


def _seam_bowler_rows(name):
    r = raw()
    return r[(r["BowlerName"].str.upper() == name) & (r["DeliveryType"] == "Seam")]


def test_bowler_formulas_match_pages():
    name = "BINURA FERNANDO"
    g = _seam_bowler_rows(name)
    runs, wk, balls = g["Runs"].sum(), (g["Wicket"] == True).sum(), len(g)
    s = M.profile(prep(parity=True), name, "bowler", {"delivery_class": "Seam"}, ["phase"])["summary"]
    assert s["Balls"] == balls and s["Wkts"] == wk
    approx(s["Econ"], runs / balls * 6); approx(s["BowlAvg"], runs / wk); approx(s["BowlSR"], balls / wk)
    approx(s["Dot%"], (g["Runs"] == 0).mean() * 100); approx(s["Bdry%"], g["Runs"].isin([4, 6]).mean() * 100)
    approx(s["Hit%"], hitting_stumps_mask(g).mean() * 100)


def test_batter_formulas_match_pages():
    name = "KUSAL MENDIS"
    r = raw()
    g = r[(r["BatsmanName"].str.upper() == name) & (r["DeliveryType"] == "Seam")]
    s = M.profile(prep(parity=True), name, "batter", {"delivery_class": "Seam"}, ["phase"])["summary"]
    assert s["Balls"] == len(g)
    approx(s["SR"], g["Runs"].sum() / len(g) * 100)
    wk = (g["Wicket"] == True).sum()
    approx(s["BatAvg"], g["Runs"].sum() / wk if wk else None) if wk else None


def test_length_percentages_match_pages():
    name = "BINURA FERNANDO"
    g = _seam_bowler_rows(name)
    cfg = resolve_format("men_t20i")
    page = pitch_bin_percentages(g, get_pitch_bins("Seam", cfg), "BounceX")
    d = prep(parity=True).df
    d = d[(d["Bowler"] == name) & (d["DeliveryClass"] == "Seam")]
    total = d["ok_length"].sum()
    mine = {k: round(100 * (d["LengthBin"] == k).sum() / total) for k in page}
    assert mine == {k: int(v) for k, v in page.items()}, (mine, page)


# ---------------------------------------------------------------------------------------------
# documented differences from the pages (default cleaning mode)
# ---------------------------------------------------------------------------------------------
def test_default_mode_documented_differences():
    base, par = prep(), prep(parity=True)
    assert base.report["untracked_balls"] > 0
    assert (base.df["DeliveryClass"] == "Seam").sum() == (par.df["DeliveryClass"] == "Seam").sum() + base.report["medium_balls_counted_as_seam"]
    # untracked balls keep runs/wickets/balls ...
    assert len(base.df) == len(par.df) and base.df["Runs"].sum() == par.df["Runs"].sum() and base.df["IsWicket"].sum() == par.df["IsWicket"].sum()
    # ... but are excluded from trajectory metrics
    ut = base.df[base.df["Untracked"]]
    assert not ut[["ok_length", "ok_speed", "ok_stumps", "ok_crease", "has_contact"]].any().any()
    assert base.df["HitV"].mean() < par.df["HitV"].mean()


def test_untracked_not_counted_as_hits():
    d = prep().df
    assert d.loc[d["Untracked"], "HitsStumps"].sum() == 0


# ---------------------------------------------------------------------------------------------
# conventions & behaviours
# ---------------------------------------------------------------------------------------------
def test_names_and_resolution():
    p = prep()
    assert M.resolve_player(p, "binura fernando", "bowler") == "BINURA FERNANDO"
    assert M.resolve_player(p, "  Kusal   Mendis ") == "KUSAL MENDIS"
    try:
        M.resolve_player(p, "Mendis")
        raise AssertionError("expected clarification")
    except M.NeedsClarification as e:
        assert len(e.candidates) > 1


def test_merge_with_is_explicit_only():
    p = prep()
    a = M.profile(p, "Hasaranga", "bowler", None, ["phase"])
    assert a.get("possible_same_player_NOT_merged")
    b = M.profile(p, "Hasaranga", "bowler", None, ["phase"], merge_with=["Wanidu Hasaranga"])
    assert b["summary"]["Balls"] > a["summary"]["Balls"] and "WANIDU HASARANGA" in b["entity"]


def test_spin_type_signs():
    p = prep(); t = p.bowlers["SpinType"]
    assert t.get("WANIDU HASARANGA") == "Leg-spin"
    assert t.get("RAMESH MENDIS") == "Off-spin"
    assert t.get("SAMIT PATEL") == "Left-arm orthodox"
    assert t.get("MAHEESH THEEKSHANA") == "Mixed (off/leg)"   # carrom/mystery: no consistent direction


def test_suspect_delivery_type_flagged_not_changed():
    p = prep()
    names = {x["bowler"] for x in p.report["suspect_delivery_type"]}
    assert "HASARANGA" in names
    assert p.df.loc[p.df["Bowler"] == "HASARANGA", "DeliveryClass"].eq("Seam").all()
    fixed = prep(fix_suspect_class=True)
    assert fixed.df.loc[fixed.df["Bowler"] == "HASARANGA", "DeliveryClass"].eq("Spin").all()


def test_sort_and_flags():
    d = prep().df
    assert d["Over"].min() >= 1 and d["Ball"].min() >= 1
    assert (d.groupby(["Match", "Innings"])["Over"].apply(lambda s: s.is_monotonic_increasing)).all()


def test_leaderboard_min_balls_and_confidence():
    lb = M.leaderboard(prep(), "bowler", "economy", {"phase": "Death"}, 60, 5, None)
    assert all(r["Balls"] >= 60 for r in lb["rows"])
    assert M.confidence(5)["label"] == "Very low" and M.confidence(400)["label"] == "High"
    assert M.confidence(5)["warnings"]


def test_zero_wickets_gives_none():
    d = prep().df
    one = d[(d["Bowler"] == "BINURA FERNANDO")].head(3)
    s = M.summary_row(one.assign(IsWicket=False), "bowler")
    assert s["BowlAvg"] is None and s["BowlSR"] is None


def test_filters():
    p = prep()
    d, _ = M.apply_filters(p, {"phase": "death", "bowler_hand": "left", "delivery_class": "pace", "years": [2024]})
    assert (d["Phase"] == "Death (17-20)").all() and (d["BowlerHand"] == "Left").all() and (d["DeliveryClass"] == "Seam").all() and (d["Year"] == 2024).all()
    try:
        M.apply_filters(p, {"grounds": ["nowhere-ville"]})
        raise AssertionError("expected clarification")
    except M.NeedsClarification:
        pass


def test_all_tools_run():
    p, st = prep(), {}
    calls = [("dataset_overview", {}), ("search_players", {"query": "noor"}),
             ("player_profile", {"player": "Noor Ahmad", "role": "bowler", "charts": ["pitch_map", "beehive"]}),
             ("player_profile", {"player": "Kusal Mendis", "role": "batter", "charts": ["wagon_wheel"]}),
             ("group_stats", {"role": "bowler", "group_by": ["phase", "batter_hand"], "chart_metric": "Econ"}),
             ("leaderboard", {"role": "batter", "metric": "strike_rate", "filters": {"delivery_class": "Spin"}}),
             ("compare_players", {"players": ["Noor Ahmad", "Samit Patel"], "role": "bowler"}),
             ("matchup", {"bowler": "Binura Fernando", "batter": "Kusal Mendis", "charts": ["pitch_map"]}),
             ("ground_report", {"ground": "premadasa"}),
             ("make_chart", {"chart": "speed_hist", "role": "bowler", "player": "Binura Fernando"}),
             ("make_chart", {"chart": "bar", "role": "bowler", "group_by": "phase", "metric": "Econ"}),
             ("make_chart", {"chart": "trend", "role": "bowler", "player": "Noor Ahmad", "group_by": "year", "metric": "Econ"})]
    for name, args in calls:
        out = T.execute(name, args, p, st)
        assert "error" not in out.result, (name, out.result)
        json.dumps(out.result, default=str)


# ---------------------------------------------------------------------------------------------
# LLM layer (mock + fake HTTP; no network)
# ---------------------------------------------------------------------------------------------
class MockLLM:
    """Plays a Groq model: first asks for a tool, then writes an answer from the tool result."""
    def __init__(self, invent_first=False):
        self.invent_first, self.n_final = invent_first, 0

    def chat(self, messages, tools=None, temperature=0.1):
        last = messages[-1]
        if last["role"] == "user" and tools and "not in the function results" not in last["content"]:
            return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "player_profile", "arguments": json.dumps({"player": "Binura Fernando", "role": "bowler", "filters": {"phase": "Death"}, "charts": ["pitch_map"]})}}]}
        if last["role"] == "tool":
            res = json.loads(last["content"])
            s = res["summary"]
            if self.invent_first:
                return {"role": "assistant", "content": f"Economy {s['Econ']} but strike rate 7.77 over {s['Balls']} balls."}
            return {"role": "assistant", "content": f"Economy {s['Econ']} over {s['Balls']} balls ({res['confidence']['label']} confidence)."}
        s = json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])["summary"]
        return {"role": "assistant", "content": f"Economy {s['Econ']} over {s['Balls']} balls."}


def test_run_turn_end_to_end():
    p, st = prep(), {}
    t = run_turn(MockLLM(), p, "Binura death overs?", [], st)
    assert t.calls[0]["tool"] == "player_profile" and t.figures and t.tables and not t.unverified
    assert isinstance(t.figures[0][1], bytes) and t.figures[0][1][:4] == b"\x89PNG"      # rendered once as PNG
    assert set(t.timing) == {"total_s", "model_s", "model_calls", "calc_s"}
    assert st["last_entity"] == "BINURA FERNANDO" and t.confidence


def test_number_guard_flags_then_repairs():
    from dansh_ai import config as C
    p = prep()
    t = run_turn(MockLLM(invent_first=True), p, "q", [], {})
    assert "7.77" in t.unverified and t.timing["model_calls"] == 2        # default: flagged, no extra model call
    C.REPAIR_PASS = True
    try:
        t = run_turn(MockLLM(invent_first=True), p, "q", [], {})
        assert "7.77" not in t.text and not t.unverified and t.timing["model_calls"] == 3
    finally:
        C.REPAIR_PASS = False
    assert unverified_numbers("strike rate 7.77", [{"a": 1.5}]) == ["7.77"]
    assert unverified_numbers("economy 8.5", [{"Econ": 8.49}]) == []   # rounding tolerant


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._b, self.headers, self.text = status, body, headers or {}, json.dumps(body)

    def json(self):
        return self._b


class _Sess:
    def __init__(self, resps):
        self.resps, self.calls = list(resps), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.resps.pop(0)


OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


def test_groq_client_request_shape_and_retry():
    import time as _t
    _t.sleep = lambda *_: None
    s = _Sess([_Resp(429, {"error": "rate"}, {"retry-after": "0"}), _Resp(200, OK)])
    c = GroqClient("gsk_x", session=s)
    msg = c.chat([{"role": "user", "content": "x"}], tools=T.TOOLS)
    assert msg["content"] == "hi" and len(s.calls) == 2
    call = s.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions" and call["headers"]["Authorization"] == "Bearer gsk_x"
    assert call["json"]["tool_choice"] == "auto" and call["json"]["tools"][0]["type"] == "function"
    assert call["json"]["reasoning_effort"] == "low" and call["json"]["max_completion_tokens"] == 2048   # gpt-oss speed settings


def test_groq_client_drops_unsupported_speed_settings():
    rej = _Resp(400, {"error": {"message": "reasoning_effort is not supported with this model"}})
    s = _Sess([rej, _Resp(200, OK), _Resp(200, OK)])
    c = GroqClient("gsk_x", model="openai/gpt-oss-120b", session=s)
    c.chat([{"role": "user", "content": "x"}]); c.chat([{"role": "user", "content": "y"}])
    assert "reasoning_effort" in s.calls[0]["json"] and "reasoning_effort" not in s.calls[1]["json"] and "reasoning_effort" not in s.calls[2]["json"]


def test_groq_client_falls_back_on_bad_tool_call():
    bad = _Resp(400, {"error": {"code": "tool_use_failed", "failed_generation": "..."}})
    s = _Sess([bad, bad, _Resp(200, OK)])
    c = GroqClient("gsk_x", model="primary", fallback_model="backup", session=s)
    c.chat([{"role": "user", "content": "x"}], tools=T.TOOLS)
    assert [x["json"]["model"] for x in s.calls] == ["primary", "primary", "backup"]


def test_groq_client_hard_error():
    s = _Sess([_Resp(401, {"error": "bad key"})])
    try:
        GroqClient("bad", session=s).chat([{"role": "user", "content": "x"}])
        raise AssertionError("expected error")
    except GroqError as e:
        assert "401" in str(e)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"PASS  {n}")
        except Exception:
            failed += 1; print(f"FAIL  {n}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

"""Groq client + the plan -> calculate -> interpret loop.

Groq exposes an OpenAI-compatible Chat Completions endpoint with tool calling. The model only PLANS (chooses a
function + arguments) and INTERPRETS (writes the answer from returned numbers). All maths is in metrics.py.
"""
from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from . import config as C
from .prep import Prepared
from .tools import TOOLS, ToolOutput, execute, shrink_json


class GroqError(Exception):
    pass


class GroqHTTPError(GroqError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Groq API error {status}: {body[:300]}")
        self.status, self.body = status, body


class GroqClient:
    def __init__(self, api_key: str, model: str = C.DEFAULT_MODEL, fallback_model: Optional[str] = C.FALLBACK_MODEL,
                 base_url: str = C.GROQ_BASE_URL, session: Optional[requests.Session] = None):
        self.api_key, self.model, self.fallback_model = api_key, model, fallback_model
        self.base_url, self.session = base_url.rstrip("/"), session or requests.Session()
        self.on_status: Optional[Callable[[str], None]] = None   # UI hook: shown while waiting on rate limits
        self.use_extras = True                                   # reasoning_effort / max_completion_tokens

    def _post(self, payload: Dict) -> Dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(3):
            r = self.session.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                try:
                    wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                except ValueError:
                    wait = 2.0 * (attempt + 1)
                if self.on_status:
                    self.on_status(f"Groq is busy ({r.status_code}) - waiting {min(wait, 20.0):.0f}s, retry {attempt + 1}/3...")
                time.sleep(min(wait, 20.0))
                continue
            raise GroqHTTPError(r.status_code, r.text)
        raise GroqError("Groq is rate-limiting or unavailable right now - please try again in a moment.")

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None, temperature: float = C.LLM_TEMPERATURE) -> Dict:
        """Returns the assistant message dict. Retries malformed tool calls, then falls back to a second model."""
        attempts = [(self.model, temperature), (self.model, 0.0)]
        if self.fallback_model and self.fallback_model != self.model:
            attempts.append((self.fallback_model, 0.0))
        last: Optional[Exception] = None
        for model, temp in attempts:
            payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temp}
            if self.use_extras:
                payload["max_completion_tokens"] = C.MAX_COMPLETION_TOKENS
                if C.REASONING_EFFORT and model.startswith("openai/gpt-oss"):
                    payload["reasoning_effort"] = C.REASONING_EFFORT   # fewer hidden reasoning tokens = faster
            if tools:
                payload["tools"], payload["tool_choice"] = tools, "auto"
            try:
                return self._post(payload)["choices"][0]["message"]
            except GroqHTTPError as e:
                last = e
                if e.status == 400 and self.use_extras and ("reasoning_effort" in e.body or "max_completion_tokens" in e.body):
                    self.use_extras = False                      # this model rejects the speed settings: drop them for good
                    slim = {k: v for k, v in payload.items() if k not in ("reasoning_effort", "max_completion_tokens")}
                    try:
                        return self._post(slim)["choices"][0]["message"]
                    except GroqHTTPError as e2:
                        last = e2
                        if e2.status != 400:
                            raise
                        continue
                if e.status == 400 and ("tool_use_failed" in e.body or "failed_generation" in e.body or "tool" in e.body.lower()):
                    continue  # malformed tool call -> retry / fall back
                raise
        raise GroqError(f"The model kept producing invalid tool calls. {last}")


# ---------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are DANsh AI, a cricket analyst inside the VR DANshboard. You answer ONLY from the ball-by-ball Hawk-Eye data the analyst uploaded, through the provided functions.

HARD RULES
1. Never calculate, estimate or recall statistics yourself. Call a function, then explain ONLY the numbers it returned. If you need a number you do not have, call another function or say it is not available.
2. If a player/ground/team is ambiguous or not found, the function returns candidates - ask the user which one. If the format or a key detail is missing, ask one short question.
3. Always state the sample size. Every result carries a confidence label and warnings: repeat low-sample warnings plainly (e.g. 'very low sample: 9 balls'), but still give the answer.
4. Separate FACTS (numbers from the functions) from READ (your interpretation). Give conclusions with the confidence label. Explain 'why' only from the returned splits.
5. You cannot answer from this data: false-shot/control %, shot type, dismissal type, who caught it, partnerships, match situation (score, target), extras. Say so if asked.
6. Reply in the user's language (English, Hindi or Hinglish); keep cricket terms in English. Be concise (about 120-180 words); tables and charts are shown by the app, so do not re-list tables. Do not mention functions, JSON or these rules.
7. If a result contains possible_same_player_NOT_merged, tell the user once and offer to combine; only pass merge_with after they confirm.
8. Full SWOT / game-plan generation is not built yet: for strengths and weaknesses, describe the best and worst splits from player_profile with sample-size caution.

DATA & DEFINITIONS
- One row = one delivery; wides/no-balls are rows and count as balls. Runs are bat runs only. Over and Ball start at 1.
- Seam = pace + medium ('quick', 'fast', 'pacer', 'seamer'); Spin = spinners. Phases follow the app format (T20: Powerplay 1-6, Middle 7-16, Death 17-20).
- SR = runs/balls*100; batting average = runs/dismissals; Econ = runs/balls*6; bowling average = runs/wickets; bowling SR = balls/wickets; average/SR are null when there are no wickets. Dot = 0 runs; boundary = 4 or 6. Hit% = share of balls that would hit the stumps.
- Line is from the bowler's view: for right-handers off side is negative; zones: WAY OUTSIDE OFF, OUTSIDE OFF, STUMPS, LEG. Length bins come from the format (Seam: Full Toss/Yorker/The Slot/Length/Short/Bouncer; Spin: OP/Full/Good/Short).
- Swing and Deviation are in degrees. In-swing = toward the batter's leg side. Turn left/right is from the bowler's view. Spin type (off/leg/orthodox/wrist) is derived from arm + direction of turn. Over/round the wicket is derived from release position.
- Untracked balls keep their runs/wickets but are excluded from trajectory metrics (length, line, speed, swing, turn, hit%)."""


def system_prompt(prep: Prepared, state: Dict) -> str:
    r = prep.report
    ctx = (f"\n\nSESSION: format={prep.cfg.label}; speed unit={prep.unit}; {r['rows']:,} deliveries, {r['matches']} matches, years {r['years']}."
           f" Active context from earlier in this chat: {json.dumps(state, default=str)}")
    return SYSTEM_PROMPT + ctx


# ---------------------------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _collect(obj, acc: List[float]):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)


def unverified_numbers(answer: str, results: List[Dict]) -> List[str]:
    """Numbers in the answer that cannot be traced to any tool result (rounding-tolerant)."""
    vals: List[float] = []
    for r in results:
        _collect(r, vals)
    text = " ".join(json.dumps(r, default=str) for r in results)
    seen_tokens = set(_NUM.findall(text))
    bad: List[str] = []
    for tok in _NUM.findall(answer):
        x = float(tok)
        if 1900 <= abs(x) <= 2100 or (x == int(x) and abs(x) <= 20):
            continue
        if tok in seen_tokens:
            continue
        dec = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if any(abs(x - v) <= tol or abs(x - v * 100) <= tol for v in vals):
            continue
        bad.append(tok)
    return list(dict.fromkeys(bad))


@dataclass
class Turn:
    text: str
    figures: List = field(default_factory=list)
    tables: List = field(default_factory=list)
    calls: List[Dict] = field(default_factory=list)
    confidence: List[Dict] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    unverified: List[str] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)


def fig_to_png(fig) -> bytes:
    """Render once; the page shows PNG bytes so old charts are never re-drawn on later reruns."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def run_turn(client, prep: Prepared, user_text: str, history: List[Dict], state: Dict,
             on_status: Optional[Callable[[str], None]] = None) -> Turn:
    msgs: List[Dict] = [{"role": "system", "content": system_prompt(prep, state)}]
    msgs += history[-C.HISTORY_TURNS:] + [{"role": "user", "content": user_text}]
    turn, results = Turn(text=""), []
    final = None
    t_start, t_llm, t_py, n_llm = time.perf_counter(), 0.0, 0.0, 0

    def llm(m, tools):
        nonlocal t_llm, n_llm
        t0 = time.perf_counter()
        try:
            return client.chat(m, tools=tools)
        finally:
            t_llm += time.perf_counter() - t0
            n_llm += 1

    for _ in range(C.MAX_TOOL_STEPS):
        if on_status:
            on_status("Planning..." if n_llm == 0 else "Writing the answer...")
        msg = llm(msgs, TOOLS)
        calls = msg.get("tool_calls") or []
        if not calls:
            final = msg.get("content") or ""
            break
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            if on_status:
                on_status(f"Calculating: {fn}")
            t0 = time.perf_counter()
            out: ToolOutput = execute(fn, args, prep, state) if isinstance(args, dict) else ToolOutput({"error": "arguments were not valid JSON"})
            t_py += time.perf_counter() - t0
            results.append(out.result)
            turn.calls.append({"tool": fn, "args": args})
            turn.figures += out.figures
            turn.tables += out.tables
            if isinstance(out.result.get("confidence"), dict):
                turn.confidence.append({"label": fn, **out.result["confidence"]})
            turn.filters += [x for x in out.result.get("filters_applied", []) if x not in turn.filters]
            msgs.append({"role": "tool", "tool_call_id": call.get("id", fn), "name": fn, "content": shrink_json(out.result)})
    if final is None:  # ran out of steps: force a written answer from what we have
        msgs.append({"role": "user", "content": "Write the answer now from the results above. Do not call more functions."})
        final = llm(msgs, None).get("content") or ""
    bad = unverified_numbers(final, results) if results else []
    if bad and C.REPAIR_PASS:  # optional extra model call: only numbers from the results
        msgs.append({"role": "assistant", "content": final})
        msgs.append({"role": "user", "content": f"These numbers are not in the function results: {', '.join(bad)}. Rewrite the answer using ONLY numbers from the results (drop anything you cannot support)."})
        final = llm(msgs, None).get("content") or final
        bad = unverified_numbers(final, results)
    t0 = time.perf_counter()
    turn.figures = [(cap, fig_to_png(f)) for cap, f in turn.figures]
    t_py += time.perf_counter() - t0
    turn.text, turn.unverified = final.strip(), bad
    turn.timing = {"total_s": round(time.perf_counter() - t_start, 1), "model_s": round(t_llm, 1), "model_calls": n_llm, "calc_s": round(t_py, 2)}
    return turn

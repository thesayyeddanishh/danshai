"""DANsh AI - chat analyst page (5th page). Reads the CSV loaded on Home; the 4 other pages are untouched."""
from __future__ import annotations

import os

import streamlit as st

from cricket_config import FORMAT_KEYS, FORMAT_LABELS
from dansh_ai import config as C
from dansh_ai.llm import GroqClient, GroqError, run_turn
from dansh_ai.prep import MissingColumns, prepare

st.set_page_config(page_title="DANsh AI", page_icon="hawk-logo.png", layout="wide")
st.title("DANsh AI")
st.caption("Ask in English, Hindi or Hinglish. Groq plans the question; Python calculates every number from your CSV.")

if "data_df" not in st.session_state:
    st.info("Upload your CSV on the Home page first.")
    st.stop()


def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


# ---- format must be confirmed by the analyst (no auto-guessing) -------------------------------------
file_name = st.session_state.get("file_name", "")
fmt_labels = [FORMAT_LABELS[k] for k in FORMAT_KEYS]
cur = st.session_state.get("cricket_format", "men_t20i")
c1, c2, c3 = st.columns([3, 1, 2])
with c1:
    fmt_choice = st.selectbox("Format for this analysis", fmt_labels, index=FORMAT_KEYS.index(cur) if cur in FORMAT_KEYS else 0, key="dansh_fmt")
fmt_key = FORMAT_KEYS[fmt_labels.index(fmt_choice)]
unit = st.session_state.get("speed_unit", "kph")
with c2:
    st.metric("Speed unit", unit.upper())
confirm_key = f"dansh_confirmed::{file_name}"
with c3:
    st.write("")
    if st.button("Use this format", type="primary"):
        st.session_state[confirm_key] = fmt_key
if st.session_state.get(confirm_key) != fmt_key:
    st.warning("Confirm the format above so DANsh AI uses the right length bins, speeds and phases.")
    st.stop()

with st.expander("Data options"):
    fix = st.checkbox("Treat bowlers labelled Seam who bowl at spin speeds as Spin (off by default; labels are used as supplied)", value=False, key="dansh_fix")

# ---- prepare data once per (file, format, unit, option) ---------------------------------------------------
sig = (file_name, len(st.session_state["data_df"]), fmt_key, unit, fix)
if st.session_state.get("dansh_sig") != sig:
    try:
        with st.spinner("Preparing data..."):
            st.session_state["dansh_prep"] = prepare(st.session_state["data_df"], fmt_key, unit, fix_suspect_class=fix)
    except MissingColumns as e:
        st.error(str(e)); st.stop()
    st.session_state["dansh_sig"] = sig
    st.session_state["dansh_msgs"], st.session_state["dansh_state"] = [], {}
prep = st.session_state["dansh_prep"]

with st.expander(f"Data quality - {prep.report['rows']:,} deliveries, {prep.report['matches']} matches"):
    for n in prep.report["notes"]:
        st.warning(n)
    rep = prep.report
    st.write({k: rep[k] for k in ("untracked_balls", "no_contact_balls", "overs_with_extra_deliveries", "medium_balls_counted_as_seam",
                                  "matches_with_one_innings", "name_case_variants_merged")})
    st.write("Excluded per metric (the ball still counts for runs/balls/wickets):", rep["excluded_per_metric"])
    if rep["handedness_conflicts_batters"]:
        st.write("Batters with both hands in the data (majority used):", rep["handedness_conflicts_batters"])

# ---- Groq key ------------------------------------------------------------------------------------------------
api_key = _secret("GROQ_API_KEY")
model = _secret("GROQ_MODEL", C.DEFAULT_MODEL)
if not api_key:
    st.error("Add GROQ_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud > Settings > Secrets). See DANSH_AI_SETUP.md.")
    st.stop()
client = GroqClient(api_key, model=model, fallback_model=_secret("GROQ_FALLBACK_MODEL", C.FALLBACK_MODEL) or None)

# ---- chat history ---------------------------------------------------------------------------------------------
msgs = st.session_state.setdefault("dansh_msgs", [])
state = st.session_state.setdefault("dansh_state", {})


def render_extras(m):
    for conf in m.get("confidence", [])[:3]:
        st.progress(min(conf["score_pct"], 100) / 100, text=f"Confidence: {conf['label']} ({conf['balls']} balls)")
        for w in conf.get("warnings", []):
            st.warning(w)
    for cap, png in m.get("figures", []):
        st.image(png, caption=cap)
    for title, df in m.get("tables", []):
        with st.expander(title):
            st.dataframe(df, use_container_width=True, hide_index=True)
    if m.get("filters"):
        st.caption("Filters: " + "; ".join(m["filters"]))
    if m.get("timing"):
        t = m["timing"]
        st.caption(f"Time {t['total_s']}s: model {t['model_s']}s ({t['model_calls']} calls), calculations {t['calc_s']}s")
    if m.get("unverified"):
        st.caption("Numbers not found in the calculated results: " + ", ".join(m["unverified"]))


top = st.columns([6, 1])
with top[1]:
    if st.button("Clear chat"):
        st.session_state["dansh_msgs"], st.session_state["dansh_state"] = [], {}
        st.rerun()

if not msgs:
    st.markdown("**Try:** `How does Binura Fernando bowl in the death overs?` | `Best strike rate vs spin in the middle overs` | `Hasaranga vs Kusal Mendis` | `Premadasa ground stats 2023` | `Kusal Mendis ka wagon wheel dikhao`")

for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

q = st.chat_input("Ask DANsh AI about a player, matchup, ground or phase...")
if q:
    with st.chat_message("user"):
        st.markdown(q)
    history = [{"role": x["role"], "content": x["content"]} for x in msgs]
    with st.chat_message("assistant"):
        status = st.empty()
        try:
            client.on_status = lambda s: status.caption(s)
            turn = run_turn(client, prep, q, history, state, on_status=lambda s: status.caption(s))
            status.empty()
            st.markdown(turn.text)
            m_out = {"role": "assistant", "content": turn.text, "figures": turn.figures, "tables": turn.tables,
                     "confidence": turn.confidence, "filters": turn.filters, "unverified": turn.unverified, "timing": turn.timing}
            render_extras(m_out)
            msgs += [{"role": "user", "content": q}, m_out]
        except GroqError as e:
            status.empty()
            st.error(f"{e}")

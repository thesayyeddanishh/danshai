# DANsh AI (page 5) - setup

## Add to the repo
Copy these into the repo root (nothing existing is modified):
`dansh_ai/`, `pages/05_DANSH_AI.py`, `tests/`, `.streamlit/secrets.toml.example`

## Groq key (GroqCloud, not xAI Grok)
1. Create a key at https://console.groq.com/keys
2. Local: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill `GROQ_API_KEY`.
   Streamlit Cloud: App > Settings > Secrets, paste the same lines. Never commit the real key.
3. Model ids change - check https://console.groq.com/docs/models. Default `openai/gpt-oss-120b`, fallback `llama-3.3-70b-versatile`
   (set `GROQ_MODEL` / `GROQ_FALLBACK_MODEL` in secrets to change them; no code edit needed).
4. `requests` is already installed with Streamlit; add it to requirements.txt if you want it explicit.

## How it works
You ask -> Groq picks one of 9 fixed functions + arguments -> Python computes with your formulas -> Groq writes the answer from those numbers.
Only aggregated results go to Groq (never raw ball-by-ball rows). Numbers in the answer are checked against the results; anything untraceable is flagged.

## Rules baked in (from your Q&A + checks on LPL.xlsx)
- Names upper-cased and space-normalised, no fuzzy merging. Possible same-player pairs are reported, never merged unless you say so.
- Handedness by majority. Medium = Seam. Over/Ball start at 1. Every row is a ball (wides/no-balls included). Runs are bat runs.
- Untracked balls (ReleaseSpeed < 0, trajectory all zero): kept for runs/balls/wickets, excluded from length/line/speed/swing/turn/hit%.
- Per-metric plausible ranges in `dansh_ai/config.py` (BounceX -10..18, speed 40-160 kph, swing/deviation +-15 deg ...).
- Phases, bins, speed groups come straight from `cricket_config.py`, so they cannot drift from the pages.
- Sign conventions are absolute from the bowler's view: line +right, Swing/Deviation +right; in-swing = toward batter's leg side.
- Seam/Spin labels are used as supplied; suspected mislabels (Seam but spin speeds) are listed, and can be switched to Spin in "Data options".

## Tests (no network)
`python tests/test_dansh_ai.py` (set `DANSH_TEST_DATA` to your CSV/XLSX). Includes exact parity checks against the pages' wagon-wheel,
lateral-zone, phase, length-bin, economy/SR/average and hit-stumps logic.

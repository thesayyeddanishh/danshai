"""All tunable constants for DANsh AI in one place (nothing here changes the 4 existing pages)."""
from __future__ import annotations

# ---- LLM (Groq, OpenAI-compatible endpoint) -------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"          # override with st.secrets["GROQ_MODEL"] / env GROQ_MODEL
FALLBACK_MODEL = "llama-3.3-70b-versatile"     # used if the main model keeps failing tool calls
MAX_TOOL_STEPS = 6                             # planner <-> Python round trips per question
HISTORY_TURNS = 8                              # previous user/assistant messages sent back to the model
TOOL_RESULT_MAX_CHARS = 9000                   # cap on what is sent to the model per tool result
LLM_TEMPERATURE = 0.1

# ---- Data cleaning (applied to a DERIVED copy; the uploaded data is never modified) ---------------
# Plausible physical ranges, applied PER METRIC (a bad BounceX never removes the ball's runs/wicket).
RANGES = {
    "BounceX": (-10.0, 18.0),       # your rule
    "BounceY": (-3.0, 3.0),
    "CreaseY": (-3.0, 3.0),
    "CreaseZ": (0.0, 5.0),
    "StumpsY": (-3.0, 3.0),
    "StumpsZ": (-0.2, 5.0),
    "ReleaseY": (-3.0, 3.0),
    "ReleaseZ": (0.5, 3.5),
    "ReleaseSpeed": (40.0, 160.0),  # kph (raised from 40-150 so 153-157 kph balls are kept)
    "Deviation": (-15.0, 15.0),     # your rule, degrees
    "Swing": (-15.0, 15.0),         # degrees
}
NO_CONTACT = -900.0                 # Interception* <= this means "no contact" (file uses -999.9)
UNTRACKED_SPEED = 0.0               # ReleaseSpeed < 0 (file uses -1.609) = ball not tracked

# ---- Dashboard definitions kept for parity -------------------------------------------------------
BOUNDARY_RUNS = (4, 6)
TURN_EPS = 0.1                      # |Deviation| <= this is "No Turn" on the Leaderboard page
CREASE_Z_MAX = 1.85                 # beehive lateral zones ignore balls above this height
MIN_BALLS_LEADERBOARD = 30          # default minimum balls for rankings (warned, never hidden)

# ---- Sample-size confidence (deterministic) --------------------------------------------------------
CONF_K = 80.0                       # score = 1 - exp(-balls / CONF_K)
CONF_BANDS = ((0.25, "Very low"), (0.50, "Low"), (0.75, "Medium"), (1.01, "High"))
MIN_WICKETS_FOR_AVG = 5             # average / strike rate rest on wickets: warn below this

# ---- Allowed filter / grouping vocabularies (the LLM is only allowed these) --------------------
GROUP_DIMS = (
    "phase", "delivery_class", "bowler_hand", "batter_hand", "length", "line", "speed_band",
    "ground", "year", "innings", "over", "bowling_team", "batting_team", "over_round",
    "spin_type", "batter", "bowler",
)

# ---- Spin-type labelling (derived from bowler arm + sign of Deviation; sign is absolute, + = bowler's right) ----
SPIN_TYPE_MIN_CONSISTENCY = 0.40    # |mean Deviation| / mean |Deviation| needed to label a bowler (else Mixed)
SPIN_TYPE_MIN_BALLS = 60            # fewer tracked spin balls than this -> "Unclassified"

import streamlit as st
import json
import csv
import os
import difflib

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_CSV       = os.path.join(_HERE, "colab", "train.csv")
_LABELS_PATH     = os.path.join(_HERE, "colab", "labeler", "labels_clean.jsonl")
_PREDICTIONS_DIR = os.path.join(_HERE, "predictions")
_RESULTS_FULL    = os.path.join(_HERE, "evaluation", "results_full.json")

CONDITIONS = ["baseline", "soft", "medium", "hard"]
CONDITION_MAP = {
    "Baseline": "baseline",
    "Soft":     "soft",
    "Medium":   "medium",
    "Hard":     "hard",
}

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="HouseLLM",
    page_icon="🩺",
    layout="wide"
)

# --- HEADER IMAGE ---
st.image("images/header.jpg", use_container_width=True)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }

    h1, h2, h3 {
        color: #5cc8ff;
    }

    section[data-testid="stSidebar"] {
        background-color: #0ea5a4;
    }

    .chat-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 20px;
    }

    .chat-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin-left: 10px;
        max-width: 75%;
        font-size: 15px;
    }

    .user {
        background-color: #1f2937;
        color: #e6edf3;
        white-space: pre-wrap;
        font-size: 13px;
    }

    .assistant {
        background-color: #0ea5a4;
        color: black;
    }

    img.avatar {
        width: 50px;
        border-radius: 50%;
    }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🩺 HouseLLM")
st.caption("A medical report generator with attitude.")

# --- SIDEBAR ---
st.sidebar.image("images/logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.header("Settings")

constraint = st.sidebar.selectbox("Constraint Type", list(CONDITION_MAP.keys()))
st.sidebar.caption(f"{constraint} Constraint Mode")

image_map = {
    "Baseline": "images/bluebg.jpg",
    "Soft":     "images/soft.png",
    "Medium":   "images/medium.jpeg",
    "Hard":     "images/team.jpg",
}
st.sidebar.image(image_map[constraint], use_container_width=True)
st.sidebar.markdown("---")

research_mode = st.sidebar.checkbox("Research Mode", value=False)
if research_mode:
    st.sidebar.caption("Raw JSON output will appear beneath each response.")

st.sidebar.markdown("---")


# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Load dialogues from train.csv and predictions from all condition JSONL files."""
    # Dialogues indexed by encounter_id
    dialogues = {}
    try:
        with open(_TRAIN_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dialogues[row["encounter_id"]] = row["dialogue"]
    except FileNotFoundError:
        pass

    # Also load from labels_clean.jsonl as fallback
    try:
        with open(_LABELS_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dialogues.setdefault(r["encounter_id"], r["dialogue"])
    except FileNotFoundError:
        pass

    # Predictions indexed by encounter_id → condition
    predictions = {}
    for condition in CONDITIONS:
        path = os.path.join(_PREDICTIONS_DIR, f"predictions_{condition}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    eid = r["encounter_id"]
                    predictions.setdefault(eid, {})[condition] = {
                        "prediction": r.get("prediction", {}),
                        "valid":      r.get("valid", False),
                    }
                except json.JSONDecodeError:
                    continue

    # Evaluation results indexed by condition → encounter_id
    evals = {}
    try:
        with open(_RESULTS_FULL, encoding="utf-8") as f:
            full_data = json.load(f)
        for condition, data in full_data.items():
            per_det   = {r["encounter_id"]: r for r in data.get("per_record_deterministic", [])}
            per_judge = {r["encounter_id"]: r for r in data.get("per_record_judge", [])}
            evals[condition] = {"deterministic": per_det, "judge": per_judge}
    except FileNotFoundError:
        pass

    return dialogues, predictions, evals


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def match_encounter(user_input: str, dialogues: dict) -> str | None:
    """Return the encounter_id that best matches the user input, or None."""
    stripped = user_input.strip()

    # Direct encounter ID match
    if stripped in dialogues:
        return stripped

    # Substring match (user pasted part of a dialogue)
    norm_input = _normalize(stripped)
    for eid, dialogue in dialogues.items():
        if norm_input in _normalize(dialogue):
            return eid

    # Fuzzy match as last resort
    norm_dialogues = {eid: _normalize(d) for eid, d in dialogues.items()}
    matches = difflib.get_close_matches(
        norm_input,
        norm_dialogues.values(),
        n=1,
        cutoff=0.5,
    )
    if matches:
        for eid, nd in norm_dialogues.items():
            if nd == matches[0]:
                return eid

    return None


# --- JSON → REPORT ---
def format_report(note: dict) -> str:
    if not note:
        return "No structured output was produced for this encounter."

    parts = []

    name = note.get("name") or "Patient"
    age  = note.get("age")
    parts.append(f"**{name}**" + (f", {age} years old" if age else ""))

    symptoms = note.get("symptoms", [])
    duration = note.get("duration")
    if symptoms:
        parts.append(
            f"**Chief Complaint:** {', '.join(symptoms)}"
            + (f" — {duration}" if duration else "")
        )

    negated = note.get("negated_symptoms", [])
    if negated:
        parts.append(f"**Denied:** {', '.join(negated)}")

    history = note.get("history", [])
    if history:
        parts.append(f"**History:** {', '.join(history)}")

    diagnosis = note.get("diagnosis", [])
    if isinstance(diagnosis, list):
        if diagnosis:
            parts.append(f"**Diagnosis:** {', '.join(diagnosis)}")
    elif diagnosis:
        parts.append(f"**Assessment:** {diagnosis}")

    treatment = note.get("treatment", [])
    if treatment:
        lines = []
        for tx in treatment:
            if isinstance(tx, dict):
                tx_type = tx.get("type", "").replace("_", " ").title()
                lines.append(f"  - *{tx_type}:* {tx.get('detail', '')}")
            else:
                lines.append(f"  - {tx}")
        parts.append("**Plan:**\n" + "\n".join(lines))

    return "\n\n".join(parts)


# --- EVALUATION DISPLAY ---
def render_evaluation(condition: str, eid: str, evals: dict):
    """Show per-patient deterministic + judge metrics for a given encounter."""
    cond_evals = evals.get(condition)
    if not cond_evals:
        return

    det   = cond_evals["deterministic"].get(eid)
    judge = cond_evals["judge"].get(eid)

    if not det and not judge:
        return

    with st.expander("Evaluation Metrics"):
        if det:
            st.markdown("**Deterministic**")
            cols = st.columns(4)
            cols[0].metric("Symptoms F1",   det.get("symptoms_f1", "—"))
            cols[1].metric("Diagnosis F1",  det.get("diagnosis_f1", "—"))
            cols[2].metric("Treatment F1",  det.get("treatment_f1", "—"))
            cols[3].metric("Hallucination", det.get("hallucination_rate", "—"))

        if judge:
            st.markdown("**LLM Judge**")
            cols = st.columns(4)
            cols[0].metric("Symptoms",   judge.get("symptoms_equivalent", "—"))
            cols[1].metric("Diagnosis",  judge.get("diagnosis_equivalent", "—"))
            cols[2].metric("Treatment",  judge.get("treatment_equivalent", "—"))
            cols[3].metric("Sym grounded", judge.get("predicted_symptoms_grounded", "—"))

            rationale = judge.get("rationale")
            if rationale:
                st.caption(f"Judge: {rationale}")


# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_eid" not in st.session_state:
    st.session_state.last_eid = None

dialogues, predictions, evals = load_data()

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-container">
            <div class="chat-bubble user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns([1, 8])
        with cols[0]:
            st.image("images/nobg.png", width=50)
        with cols[1]:
            st.markdown(msg["content"])
            if research_mode and msg.get("encounter_id") and msg.get("condition"):
                render_evaluation(msg["condition"], msg["encounter_id"], evals)
            if research_mode and msg.get("raw_json"):
                with st.expander("Raw JSON"):
                    st.json(msg["raw_json"])

# --- LIVE REPORT PANEL ---
if st.session_state.last_eid:
    eid = st.session_state.last_eid
    condition = CONDITION_MAP[constraint]
    st.divider()
    st.subheader(f"Current Report — {eid} ({constraint})")
    if condition not in predictions.get(eid, {}):
        st.info(f"No **{constraint}** prediction exists for **{eid}**.")
    else:
        pred_data = predictions[eid][condition]
        live_json = pred_data["prediction"]
        cols = st.columns([1, 8])
        with cols[0]:
            st.image("images/nobg.png", width=50)
        with cols[1]:
            st.markdown(format_report(live_json))
            if research_mode:
                render_evaluation(condition, eid, evals)
            if research_mode and live_json:
                with st.expander("Raw JSON"):
                    st.json(live_json)

# --- INPUT ---
user_input = st.chat_input("Paste a patient dialogue...")

# --- HANDLE INPUT ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    st.markdown(f"""
    <div class="chat-container">
        <div class="chat-bubble user">{user_input}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("House is thinking..."):
        condition = CONDITION_MAP[constraint]
        eid = match_encounter(user_input, dialogues)

        if eid:
            st.session_state.last_eid = eid

        if eid is None:
            report   = "I don't recognise that dialogue. Live inference for new dialogues is coming soon."
            raw_json = {}
        elif condition not in predictions.get(eid, {}):
            report   = (
                f"Matched **{eid}** but no **{constraint}** prediction exists yet. "
                f"Run `run_inference.py` on Colab with `--conditions {condition}` to generate it."
            )
            raw_json = {}
        else:
            pred_data = predictions[eid][condition]
            raw_json  = pred_data["prediction"]
            report    = format_report(raw_json)

    cols = st.columns([1, 8])
    with cols[0]:
        st.image("images/nobg.png", width=50)
    with cols[1]:
        st.markdown(report)
        if research_mode and eid:
            render_evaluation(condition, eid, evals)
        if research_mode and raw_json:
            with st.expander("Raw JSON"):
                st.json(raw_json)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    report,
        "raw_json":   raw_json,
        "encounter_id": eid,
        "condition":  condition,
    })

    st.rerun()

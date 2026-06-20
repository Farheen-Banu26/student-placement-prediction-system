"""
app.py
------
Streamlit Web Application for the Student Placement Prediction System.
Features:
  - User authentication (signup/login) before prediction access
  - Input form for all 22 student features
  - Placement prediction (Placed / Not Placed) with probability %
  - Personalised improvement suggestions
  - Prediction history display
  - Model accuracy badge
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from auth import init_session, is_logged_in, login_user, logout_user, signup_user
from database import get_user_predictions, store_prediction

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎓 Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS  — fixed all input / selectbox visibility issues
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ──────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #f0f0f0;
}

/* ── Card ─────────────────────────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}

/* ── Hero banner ─────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 40px 20px 20px;
    background: linear-gradient(135deg,rgba(102,126,234,.25),rgba(118,75,162,.25));
    border-radius: 20px;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,.1);
}
.hero h1 { font-size: 2.6rem; font-weight: 800; color: #fff !important; margin-bottom: 8px; }
.hero p  { font-size: 1.1rem; color: #c9bcff; }

/* ── Result badges ───────────────────────────────────────────────────── */
.badge-placed {
    display: inline-block;
    background: linear-gradient(135deg,#11998e,#38ef7d);
    color: #fff; font-size: 1.6rem; font-weight: 700;
    padding: 12px 36px; border-radius: 50px; margin: 10px 0;
    text-align: center; box-shadow: 0 4px 20px rgba(56,239,125,.35);
}
.badge-notplaced {
    display: inline-block;
    background: linear-gradient(135deg,#c0392b,#f05454);
    color: #fff; font-size: 1.6rem; font-weight: 700;
    padding: 12px 36px; border-radius: 50px; margin: 10px 0;
    text-align: center; box-shadow: 0 4px 20px rgba(240,84,84,.35);
}

/* ── Accuracy pill ───────────────────────────────────────────────────── */
.accuracy-pill {
    display: inline-block;
    background: linear-gradient(135deg,#667eea,#764ba2);
    color: #fff; font-size: 1rem; font-weight: 600;
    padding: 6px 20px; border-radius: 50px;
}

/* ─────────────────────────────────────────────────────────────────────
   SIDEBAR
────────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,.95) !important;
    border-right: 1px solid rgba(255,255,255,.08);
}
/* Force ALL text inside sidebar to be light */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] code,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #e0d8ff !important;
}

/* ─────────────────────────────────────────────────────────────────────
   LABELS  (global)
────────────────────────────────────────────────────────────────────── */
label,
.stTextInput    > label,
.stNumberInput  > label,
.stSelectbox    > label,
.stTextArea     > label,
.stSlider       > label {
    color: #d0c9ff !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}

/* ─────────────────────────────────────────────────────────────────────
   TEXT  /  NUMBER  /  TEXTAREA  INPUTS
────────────────────────────────────────────────────────────────────── */
.stTextInput   > div > div > input,
.stNumberInput > div > div > input,
.stTextArea    > div > div > textarea {
    background-color: #1e1a3d !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(102,126,234,.6) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: .95rem !important;
    caret-color: #ffffff;
}
.stTextInput   > div > div > input:hover,
.stNumberInput > div > div > input:hover,
.stTextArea    > div > div > textarea:hover {
    border-color: rgba(102,126,234,.9) !important;
}
.stTextInput   > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea    > div > div > textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,.25) !important;
    outline: none !important;
}
/* Placeholder */
.stTextInput   > div > div > input::placeholder,
.stNumberInput > div > div > input::placeholder,
.stTextArea    > div > div > textarea::placeholder {
    color: #8a84b8 !important;
    opacity: 1 !important;
}
/* Number stepper buttons */
.stNumberInput > div > div > div > button {
    background: rgba(102,126,234,.18) !important;
    border: 1px solid rgba(102,126,234,.4) !important;
    color: #c9bcff !important;
    border-radius: 6px !important;
}
.stNumberInput > div > div > div > button:hover {
    background: rgba(102,126,234,.38) !important;
}

/* ─────────────────────────────────────────────────────────────────────
   SELECTBOX  — every layer targeted
   Streamlit wraps BaseWeb Select inside multiple divs.
   We target each one so the selected value is always white.
────────────────────────────────────────────────────────────────────── */

/* 1. Outermost Streamlit container */
.stSelectbox > div > div {
    background-color: #1e1a3d !important;
    border: 1.5px solid rgba(102,126,234,.6) !important;
    border-radius: 10px !important;
}

/* 2. BaseWeb Select root */
div[data-baseweb="select"] {
    background-color: #1e1a3d !important;
}
div[data-baseweb="select"] > div {
    background-color: #1e1a3d !important;
    border: none !important;
    border-radius: 10px !important;
}

/* 3. Every child div / span inside the select control — force white */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] p {
    color: #ffffff !important;
    background-color: #1e1a3d !important;
}

/* 4. The specific value-container children BaseWeb renders */
div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
div[data-baseweb="select"] [class*="ValueContainer"] span,
div[data-baseweb="select"] [class*="SingleValue"],
div[data-baseweb="select"] [class*="Placeholder"] {
    color: #ffffff !important;
    background-color: transparent !important;
}

/* 5. Dropdown arrow icon */
div[data-baseweb="select"] svg,
.stSelectbox svg {
    fill: #c9bcff !important;
    color: #c9bcff !important;
}

/* 6. Hover / focus ring on the control */
div[data-baseweb="select"] > div:hover {
    border-color: rgba(102,126,234,.9) !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,.25) !important;
}

/* ─────────────────────────────────────────────────────────────────────
   DROPDOWN MENU  (BaseWeb popover that floats above the page)
────────────────────────────────────────────────────────────────────── */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] > div > div {
    background-color: #1e1a3d !important;
    border: 1px solid rgba(102,126,234,.45) !important;
    border-radius: 10px !important;
}
/* Menu list */
ul[data-baseweb="menu"],
div[data-baseweb="menu"] {
    background-color: #1e1a3d !important;
}
/* Individual option items */
ul[data-baseweb="menu"] li,
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] [role="option"],
li[role="option"] {
    color: #e0d8ff !important;
    background-color: #1e1a3d !important;
    padding: 10px 14px !important;
}
ul[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] [role="option"]:hover,
li[role="option"]:hover {
    background-color: rgba(102,126,234,.28) !important;
    color: #ffffff !important;
}
/* Currently selected item in menu */
li[aria-selected="true"],
div[aria-selected="true"],
[role="option"][aria-selected="true"] {
    background-color: rgba(102,126,234,.4) !important;
    color: #ffffff !important;
}
/* Text inside list items */
ul[data-baseweb="menu"] li span,
ul[data-baseweb="menu"] li div,
div[data-baseweb="menu"] li span,
div[data-baseweb="menu"] li div {
    color: #e0d8ff !important;
}

/* ─────────────────────────────────────────────────────────────────────
   SLIDER
────────────────────────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: rgba(102,126,234,.35) !important;
}
[data-testid="stThumbValue"] { color: #c9bcff !important; }

/* ─────────────────────────────────────────────────────────────────────
   BUTTONS
────────────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100%;
    transition: all .3s ease;
}
.stButton > button:hover {
    opacity: .88;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102,126,234,.4) !important;
}
/* Form submit — green gradient to stand out */
.stFormSubmitButton > button {
    background: linear-gradient(135deg,#11998e,#38ef7d) !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    border: none !important;
    width: 100%;
    margin-top: 12px;
    transition: all .3s ease;
}
.stFormSubmitButton > button:hover {
    opacity: .9;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(56,239,125,.35) !important;
}

/* ─────────────────────────────────────────────────────────────────────
   MISC
────────────────────────────────────────────────────────────────────── */
/* Suggestion box */
.suggestion {
    background: rgba(255,193,7,.12);
    border-left: 4px solid #ffc107;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    color: #ffe082;
    font-size: .95rem;
}
/* Section headers */
h2, h3 { color: #c9bcff !important; }
hr     { border-color: rgba(255,255,255,.1) !important; }
/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #fff !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #b0a8d8 !important; }
/* Alert boxes */
.stAlert { border-radius: 10px !important; }
/* DataFrame */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ─────────────────────────────────────────────────────────────────────
   GLOBAL FALLBACK  — catch any remaining BaseWeb text that slips through
────────────────────────────────────────────────────────────────────── */
/* Any BaseWeb component whose immediate text is still dark */
[data-baseweb] { color: #f0f0f0; }
[data-baseweb] span,
[data-baseweb] div,
[data-baseweb] p { color: inherit; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Model & Feature List
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    """Load the trained ML model from disk."""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    if not os.path.exists(model_path):
        return None, None
    model    = joblib.load(model_path)
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return model, features

model, feature_cols = load_model()

# ---------------------------------------------------------------------------
# Helper — Model Accuracy
# ---------------------------------------------------------------------------
ACCURACY_FILE = os.path.join(MODEL_DIR, "model_accuracy.txt")

def get_model_accuracy() -> str:
    if os.path.exists(ACCURACY_FILE):
        with open(ACCURACY_FILE) as f:
            return f.read().strip()
    return "N/A"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎓 Navigation")
        st.markdown("---")

        if is_logged_in():
            user = st.session_state.username
            st.markdown(f"👤 **Logged in as:** `{user}`")
            st.markdown("---")
            page = st.radio(
                "Go to",
                ["🔮 Predict Placement", "📜 My History", "ℹ️ About"],
                label_visibility="collapsed",
            )
            st.markdown("---")
            if st.button("🚪 Logout"):
                logout_user()
                st.success("Logged out successfully.")
                st.rerun()
        else:
            page = st.radio(
                "Go to",
                ["🔐 Login", "📝 Sign Up"],
                label_visibility="collapsed",
            )

        acc = get_model_accuracy()
        if acc != "N/A":
            st.markdown("---")
            st.markdown("**🤖 Model Accuracy**")
            st.markdown(f'<div class="accuracy-pill">🎯 {acc}%</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Student Placement Predictor v1.0")

    return page

# ---------------------------------------------------------------------------
# Page: Login
# ---------------------------------------------------------------------------
def page_login():
    st.markdown("""
    <div class="hero">
        <h1>🎓 Student Placement Predictor</h1>
        <p>AI-powered tool to predict your campus placement outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔐 Login to Your Account")
        username = st.text_input("Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

        if st.button("Login →"):
            success, msg = login_user(username, password)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)
        st.info("Don't have an account? Switch to **Sign Up** in the sidebar.")

# ---------------------------------------------------------------------------
# Page: Sign Up
# ---------------------------------------------------------------------------
def page_signup():
    st.markdown("""
    <div class="hero">
        <h1>🎓 Student Placement Predictor</h1>
        <p>Create your free account and start predicting your placement outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Create New Account")
        username = st.text_input("Username", placeholder="Choose a unique username", key="su_user")
        email    = st.text_input("Email", placeholder="your@email.com", key="su_email")
        password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="su_pass")
        confirm  = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="su_confirm")

        if st.button("Create Account →"):
            if password != confirm:
                st.error("❌ Passwords do not match.")
            else:
                success, msg = signup_user(username, email, password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)
        st.info("Already have an account? Switch to **Login** in the sidebar.")

# ---------------------------------------------------------------------------
# Page: Predict Placement
# ---------------------------------------------------------------------------
def page_predict():
    st.markdown("""
    <div class="hero">
        <h1>🔮 Placement Prediction</h1>
        <p>Fill in the form below to predict your campus placement outcome instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error(
            "⚠️ Model not found! Please run **StudentPlacementPrediction.ipynb** first "
            "to train and save `model.pkl` and `features.pkl`."
        )
        return

    with st.form("prediction_form"):
        st.subheader("📋 Student Profile Information")

        # Row 1 — Personal Info
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age    = st.number_input("Age", min_value=17, max_value=30, value=21)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c3:
            cgpa   = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01,
                                      help="Cumulative Grade Point Average (0 – 10)")
        with c4:
            branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"])

        # Row 2 — Academic Profile
        st.markdown("---")
        st.markdown("**📚 Academic & Co-curricular Profile**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            college_tier         = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
        with c2:
            internships_count    = st.number_input("Internships Done", min_value=0, max_value=10, value=1)
        with c3:
            projects_count       = st.number_input("Projects Done", min_value=0, max_value=20, value=2)
        with c4:
            certifications_count = st.number_input("Certifications", min_value=0, max_value=20, value=2)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            hackathons_participated = st.number_input("Hackathons", min_value=0, max_value=20, value=1)
        with c2:
            github_repos            = st.number_input("GitHub Repos", min_value=0, max_value=50, value=3)
        with c3:
            linkedin_connections    = st.number_input("LinkedIn Connections", min_value=0, max_value=2000, value=300)
        with c4:
            backlogs                = st.number_input("Backlogs", min_value=0, max_value=10, value=0)

        # Row 3 — Skill Scores
        st.markdown("---")
        st.markdown("**⚡ Skill Scores (0 – 100)**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            coding_skill_score        = st.slider("Coding Skill",      0.0, 100.0, 65.0)
        with c2:
            aptitude_score            = st.slider("Aptitude",          0.0, 100.0, 65.0)
        with c3:
            communication_skill_score = st.slider("Communication",     0.0, 100.0, 65.0)
        with c4:
            logical_reasoning_score   = st.slider("Logical Reasoning", 0.0, 100.0, 65.0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mock_interview_score  = st.slider("Mock Interview",   0.0, 100.0, 65.0)
        with c2:
            extracurricular_score = st.slider("Extracurricular",  0.0, 100.0, 50.0)
        with c3:
            leadership_score      = st.slider("Leadership",       0.0, 100.0, 50.0)
        with c4:
            attendance_percentage = st.slider("Attendance %",     0.0, 100.0, 80.0)

        # Row 4 — Lifestyle
        st.markdown("---")
        st.markdown("**🌙 Lifestyle Factors**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            volunteer_experience = st.selectbox("Volunteer Experience", ["Yes", "No"])
        with c2:
            sleep_hours          = st.number_input("Sleep (hrs/day)", min_value=0.0, max_value=12.0,
                                                    value=7.0, step=0.5)
        with c3:
            study_hours_per_day  = st.number_input("Study (hrs/day)", min_value=0.0, max_value=12.0,
                                                     value=4.0, step=0.5)

        submit = st.form_submit_button("🔮 Predict My Placement")

    # ------------------------------------------------------------------ #
    #  Prediction Logic  (unchanged)
    # ------------------------------------------------------------------ #
    if submit:
        gender_enc    = 1 if gender == "Male" else 0
        volunteer_enc = 1 if volunteer_experience == "Yes" else 0

        college_tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
        branch_map = {"CSE": 0, "Civil": 1, "ECE": 2, "EEE": 3, "IT": 4, "Mechanical": 5}

        college_tier_enc = college_tier_map.get(college_tier, 1)
        branch_enc       = branch_map.get(branch, 0)

        input_data = {
            "age":                        age,
            "gender":                     gender_enc,
            "cgpa":                       cgpa,
            "branch":                     branch_enc,
            "college_tier":               college_tier_enc,
            "internships_count":          internships_count,
            "projects_count":             projects_count,
            "certifications_count":       certifications_count,
            "coding_skill_score":         coding_skill_score,
            "aptitude_score":             aptitude_score,
            "communication_skill_score":  communication_skill_score,
            "logical_reasoning_score":    logical_reasoning_score,
            "hackathons_participated":    hackathons_participated,
            "github_repos":               github_repos,
            "linkedin_connections":       linkedin_connections,
            "mock_interview_score":       mock_interview_score,
            "attendance_percentage":      attendance_percentage,
            "backlogs":                   backlogs,
            "extracurricular_score":      extracurricular_score,
            "leadership_score":           leadership_score,
            "volunteer_experience":       volunteer_enc,
            "sleep_hours":                sleep_hours,
            "study_hours_per_day":        study_hours_per_day,
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        prediction_raw = model.predict(input_df)[0]
        proba          = model.predict_proba(input_df)[0]
        placed_proba   = round(proba[1] * 100, 2)
        not_placed_pr  = round(proba[0] * 100, 2)
        prediction_label = "Placed" if prediction_raw == 1 else "Not Placed"

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        col_res, col_proba = st.columns([1, 1])
        with col_res:
            if prediction_label == "Placed":
                st.markdown('<div class="badge-placed">✅ PLACED 🎉</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-notplaced">❌ NOT PLACED 😔</div>', unsafe_allow_html=True)
        with col_proba:
            st.metric("Placed Probability",     f"{placed_proba}%")
            st.metric("Not Placed Probability", f"{not_placed_pr}%")

        st.progress(int(placed_proba), text=f"Placement Probability: {placed_proba}%")

        st.markdown("---")
        st.subheader("💡 Personalised Improvement Tips")

        suggestions = []
        if coding_skill_score < 60:
            suggestions.append("💻 Improve your <b>Coding Skill Score</b> — practice on LeetCode / HackerRank daily.")
        if internships_count == 0:
            suggestions.append("🏢 Try to complete at least <b>1–2 internships</b> to gain industry exposure.")
        if communication_skill_score < 60:
            suggestions.append("🗣️ Work on your <b>Communication Skills</b> — join debate clubs or public speaking sessions.")
        if cgpa < 7.0:
            suggestions.append("📚 Focus on improving your <b>CGPA</b> — aim for 7.5+ for better placement opportunities.")
        if projects_count < 2:
            suggestions.append("🔧 Build at least <b>2–3 projects</b> and host them on GitHub to strengthen your portfolio.")
        if mock_interview_score < 60:
            suggestions.append("🎤 Practice <b>Mock Interviews</b> regularly to prepare for technical and HR rounds.")
        if aptitude_score < 60:
            suggestions.append("🧠 Practice <b>Aptitude & Quantitative</b> questions — many companies test this heavily.")
        if github_repos < 2:
            suggestions.append("📁 Create and maintain a good <b>GitHub profile</b> with well-documented projects.")
        if linkedin_connections < 150:
            suggestions.append("🔗 Grow your <b>LinkedIn Network</b> — connect with professionals and alumni.")
        if backlogs > 0:
            suggestions.append(f"⚠️ Clear your <b>{backlogs} backlog(s)</b> — active backlogs reduce placement chances significantly.")
        if attendance_percentage < 75:
            suggestions.append("📅 Maintain <b>Attendance above 75%</b> — it is mandatory for most placement drives.")

        if not suggestions:
            st.success("🌟 Great profile! You have a strong chance of getting placed. Keep it up!")
        else:
            for tip in suggestions:
                st.markdown(f'<div class="suggestion">{tip}</div>', unsafe_allow_html=True)

        user_id = st.session_state.user_id
        store_prediction(
            user_id         = user_id,
            cgpa            = cgpa,
            branch          = branch,
            internships     = internships_count,
            projects        = projects_count,
            coding_score    = coding_skill_score,
            aptitude_score  = aptitude_score,
            comm_score      = communication_skill_score,
            prediction      = prediction_label,
            probability     = placed_proba,
        )
        st.caption("✔️ Prediction saved to your history.")

# ---------------------------------------------------------------------------
# Page: Prediction History
# ---------------------------------------------------------------------------
def page_history():
    st.subheader("📜 My Prediction History")
    user_id     = st.session_state.user_id
    predictions = get_user_predictions(user_id)

    if not predictions:
        st.info("You have no predictions yet. Go to **Predict Placement** to get started!")
        return

    df = pd.DataFrame(predictions)
    df.rename(columns={
        "cgpa": "CGPA", "branch": "Branch", "internships": "Internships",
        "projects": "Projects", "coding_score": "Coding Score",
        "aptitude_score": "Aptitude", "comm_score": "Communication",
        "prediction": "Result", "probability": "Prob %", "predicted_at": "Date & Time",
    }, inplace=True)

    def colour_result(val):
        colour = "#38ef7d" if val == "Placed" else "#f05454"
        return f"color: {colour}; font-weight: bold"

    styled = df.style.applymap(colour_result, subset=["Result"])
    st.dataframe(styled, use_container_width=True)

    total      = len(df)
    placed     = (df["Result"] == "Placed").sum()
    not_placed = total - placed
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total)
    col2.metric("Placed Predictions", placed)
    col3.metric("Not Placed", not_placed)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    **Student Placement Prediction System** is an AI-powered web application that
    predicts whether a student will get placed based on their academic background,
    skill scores, and extracurricular activities.

    ---
    ### 🛠️ Tech Stack
    | Component       | Technology               |
    |-----------------|--------------------------|
    | Frontend        | Streamlit                |
    | ML Models       | Scikit-learn             |
    | Database        | SQLite (student.db)      |
    | Auth            | bcrypt password hashing  |
    | Data Processing | Pandas, NumPy            |
    | Visualisation   | Matplotlib, Seaborn      |

    ---
    ### 🤖 Models Trained
    - **Logistic Regression** — baseline linear classifier
    - **Decision Tree Classifier** — interpretable rule-based model
    - **Random Forest Classifier** — ensemble model (best performer)

    ---
    ### 📊 Dataset
    - **Rows:** 100,000 student records
    - **Target:** `placement_status` (Placed / Not Placed)
    - **Features:** 23 features (academic, skill, lifestyle)

    ---
    ### 👩‍💻 Developed for College Mini Project Submission — 2026
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main App Router
# ---------------------------------------------------------------------------
def main():
    init_session()
    page = render_sidebar()

    if not is_logged_in():
        if "Login" in page:
            page_login()
        else:
            page_signup()
    else:
        if "Predict" in page:
            page_predict()
        elif "History" in page:
            page_history()
        elif "About" in page:
            page_about()


if __name__ == "__main__":
    main()

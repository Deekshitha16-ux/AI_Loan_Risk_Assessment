import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-Based Loan Risk Assessment", page_icon=":bank:", layout="wide")

MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "loan_model3.pkl",
]

FEATURE_ORDER = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EducationLevel",
    "Experience",
    "LoanAmount",
    "LoanDuration",
    "NumberOfDependents",
    "MonthlyDebtPayments",
    "CreditCardUtilizationRate",
    "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "DebtToIncomeRatio",
    "BankruptcyHistory",
    "PreviousLoanDefaults",
    "PaymentHistory",
    "LengthOfCreditHistory",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "TotalAssets",
    "TotalLiabilities",
    "UtilityBillsPaymentHistory",
    "JobTenure",
    "EmploymentStatus_Self-Employed",
    "EmploymentStatus_Unemployed",
    "MaritalStatus_Married",
    "MaritalStatus_Single",
    "MaritalStatus_Widowed",
    "HomeOwnershipStatus_Other",
    "HomeOwnershipStatus_Own",
    "HomeOwnershipStatus_Rent",
    "LoanPurpose_Debt Consolidation",
    "LoanPurpose_Education",
    "LoanPurpose_Home",
    "LoanPurpose_Other",
]

RAW_FEATURES = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EducationLevel",
    "Experience",
    "LoanAmount",
    "LoanDuration",
    "NumberOfDependents",
    "MonthlyDebtPayments",
    "CreditCardUtilizationRate",
    "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "DebtToIncomeRatio",
    "BankruptcyHistory",
    "PreviousLoanDefaults",
    "PaymentHistory",
    "LengthOfCreditHistory",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "TotalAssets",
    "TotalLiabilities",
    "UtilityBillsPaymentHistory",
    "JobTenure",
    "EmploymentStatus",
    "MaritalStatus",
    "HomeOwnershipStatus",
    "LoanPurpose",
]

CATEGORICAL_COLUMNS = ["EmploymentStatus", "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"]
EDUCATION_MAP = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}


def encode_input(input_data: dict) -> pd.DataFrame:
    normalized_input = dict(input_data)
    if "EducationLevel" in normalized_input and isinstance(normalized_input["EducationLevel"], str):
        normalized_input["EducationLevel"] = EDUCATION_MAP[normalized_input["EducationLevel"]]

    raw_input = pd.DataFrame([normalized_input], columns=RAW_FEATURES)
    encoded = pd.get_dummies(raw_input, columns=CATEGORICAL_COLUMNS, drop_first=True).astype(float)
    return encoded.reindex(columns=FEATURE_ORDER, fill_value=0.0)


def get_model_display_name(model) -> str:
    model_name = model.__class__.__name__
    display_names = {
        "GradientBoostingClassifier": "Gradient Boosting Classifier",
        "RandomForestClassifier": "Random Forest Classifier",
        "LogisticRegression": "Logistic Regression",
    }
    return display_names.get(model_name, model_name)


@st.cache_resource
def load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            with model_path.open("rb") as file:
                return pickle.load(file)
    raise FileNotFoundError("No trained model file found. Please place loan_model3.pkl in the project folder.")


model = load_model()

THEMES = {
    "Light": {
        "app_bg": "radial-gradient(1200px 560px at 12% -12%, rgba(20, 184, 166, 0.20), transparent 62%), radial-gradient(980px 520px at 88% -16%, rgba(245, 158, 11, 0.16), transparent 64%), linear-gradient(180deg, #f4f9fb 0%, #f7fbff 46%, #ffffff 100%)",
        "app_pattern": "radial-gradient(circle at 24% 30%, rgba(15, 118, 110, 0.10) 0 80px, transparent 81px), radial-gradient(circle at 80% 65%, rgba(37, 99, 235, 0.07) 0 110px, transparent 111px)",
        "sidebar_bg": "linear-gradient(180deg, #0b3935 0%, #123f4a 100%)",
        "panel": "#ffffff",
        "panel_soft": "#f8fafc",
        "text": "#111827",
        "muted": "#667085",
        "line": "#dbe4ee",
        "brand": "#0f766e",
        "brand_2": "#16a34a",
        "accent": "#f59e0b",
        "shadow": "rgba(17, 24, 39, 0.07)",
        "sidebar_text": "#eefdfb",
    },
    "Bright": {
        "app_bg": "radial-gradient(1150px 620px at 8% -12%, rgba(234, 88, 12, 0.22), transparent 60%), radial-gradient(900px 520px at 94% -8%, rgba(219, 39, 119, 0.18), transparent 58%), linear-gradient(180deg, #fff4e8 0%, #fff1f2 38%, #eef6ff 72%, #ffffff 100%)",
        "app_pattern": "radial-gradient(circle at 20% 75%, rgba(234, 88, 12, 0.10) 0 88px, transparent 89px), radial-gradient(circle at 78% 26%, rgba(219, 39, 119, 0.09) 0 120px, transparent 121px)",
        "sidebar_bg": "linear-gradient(180deg, #7c2d12 0%, #be123c 100%)",
        "panel": "#ffffff",
        "panel_soft": "#fff7ed",
        "text": "#172033",
        "muted": "#5b6475",
        "line": "#fed7aa",
        "brand": "#ea580c",
        "brand_2": "#db2777",
        "accent": "#2563eb",
        "shadow": "rgba(234, 88, 12, 0.12)",
        "sidebar_text": "#fff7ed",
    },
    "Dark": {
        "app_bg": "radial-gradient(1200px 620px at 15% -10%, rgba(56, 189, 248, 0.18), transparent 58%), radial-gradient(1000px 560px at 90% -8%, rgba(34, 197, 94, 0.16), transparent 60%), linear-gradient(180deg, #0b1220 0%, #111827 42%, #0f172a 100%)",
        "app_pattern": "radial-gradient(circle at 18% 62%, rgba(56, 189, 248, 0.11) 0 95px, transparent 96px), radial-gradient(circle at 82% 34%, rgba(34, 197, 94, 0.10) 0 120px, transparent 121px)",
        "sidebar_bg": "linear-gradient(180deg, #030712 0%, #111827 100%)",
        "panel": "#1f2937",
        "panel_soft": "#111827",
        "text": "#f9fafb",
        "muted": "#cbd5e1",
        "line": "#374151",
        "brand": "#38bdf8",
        "brand_2": "#22c55e",
        "accent": "#fbbf24",
        "shadow": "rgba(0, 0, 0, 0.28)",
        "sidebar_text": "#f9fafb",
    },
    "Ocean": {
        "app_bg": "radial-gradient(1200px 620px at 15% -10%, rgba(14, 165, 233, 0.20), transparent 58%), radial-gradient(1050px 580px at 88% -8%, rgba(45, 212, 191, 0.18), transparent 60%), linear-gradient(180deg, #e0f2fe 0%, #f0f9ff 40%, #f8fafc 100%)",
        "app_pattern": "radial-gradient(circle at 20% 65%, rgba(2, 132, 199, 0.10) 0 95px, transparent 96px), radial-gradient(circle at 80% 30%, rgba(13, 148, 136, 0.08) 0 120px, transparent 121px)",
        "sidebar_bg": "linear-gradient(180deg, #0c4a6e 0%, #155e75 100%)",
        "panel": "#ffffff",
        "panel_soft": "#f0f9ff",
        "text": "#0f172a",
        "muted": "#475569",
        "line": "#bae6fd",
        "brand": "#0284c7",
        "brand_2": "#0d9488",
        "accent": "#0ea5e9",
        "shadow": "rgba(2, 132, 199, 0.12)",
        "sidebar_text": "#e0f2fe",
    },
    "Forest": {
        "app_bg": "radial-gradient(1200px 620px at 14% -12%, rgba(34, 197, 94, 0.20), transparent 60%), radial-gradient(980px 560px at 92% -8%, rgba(163, 230, 53, 0.16), transparent 58%), linear-gradient(180deg, #ecfdf5 0%, #f7fee7 42%, #f8fafc 100%)",
        "app_pattern": "radial-gradient(circle at 22% 30%, rgba(22, 163, 74, 0.09) 0 88px, transparent 89px), radial-gradient(circle at 78% 70%, rgba(132, 204, 22, 0.08) 0 115px, transparent 116px)",
        "sidebar_bg": "linear-gradient(180deg, #14532d 0%, #166534 100%)",
        "panel": "#ffffff",
        "panel_soft": "#f7fee7",
        "text": "#052e16",
        "muted": "#3f6212",
        "line": "#bbf7d0",
        "brand": "#15803d",
        "brand_2": "#65a30d",
        "accent": "#16a34a",
        "shadow": "rgba(21, 128, 61, 0.12)",
        "sidebar_text": "#ecfccb",
    },
    "Rose Gold": {
        "app_bg": "radial-gradient(1100px 620px at 10% -12%, rgba(244, 114, 182, 0.20), transparent 58%), radial-gradient(950px 540px at 92% -8%, rgba(251, 146, 60, 0.17), transparent 60%), linear-gradient(180deg, #fff1f2 0%, #fff7ed 46%, #fffbeb 100%)",
        "app_pattern": "radial-gradient(circle at 24% 70%, rgba(219, 39, 119, 0.10) 0 90px, transparent 91px), radial-gradient(circle at 78% 25%, rgba(234, 88, 12, 0.08) 0 120px, transparent 121px)",
        "sidebar_bg": "linear-gradient(180deg, #831843 0%, #be123c 100%)",
        "panel": "#ffffff",
        "panel_soft": "#fff1f2",
        "text": "#3f0720",
        "muted": "#7f1d1d",
        "line": "#fecdd3",
        "brand": "#db2777",
        "brand_2": "#ea580c",
        "accent": "#f43f5e",
        "shadow": "rgba(219, 39, 119, 0.14)",
        "sidebar_text": "#ffe4e6",
    },
    "Midnight Neon": {
        "app_bg": "radial-gradient(1180px 620px at 12% -10%, rgba(59, 130, 246, 0.22), transparent 58%), radial-gradient(980px 560px at 90% -12%, rgba(236, 72, 153, 0.20), transparent 60%), linear-gradient(180deg, #09090b 0%, #111827 48%, #0f172a 100%)",
        "app_pattern": "radial-gradient(circle at 20% 62%, rgba(59, 130, 246, 0.13) 0 100px, transparent 101px), radial-gradient(circle at 82% 30%, rgba(236, 72, 153, 0.12) 0 130px, transparent 131px)",
        "sidebar_bg": "linear-gradient(180deg, #020617 0%, #1e1b4b 100%)",
        "panel": "#111827",
        "panel_soft": "#0b1120",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "line": "#334155",
        "brand": "#3b82f6",
        "brand_2": "#ec4899",
        "accent": "#22d3ee",
        "shadow": "rgba(2, 6, 23, 0.35)",
        "sidebar_text": "#e2e8f0",
    },
}

selected_theme = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0, key="theme_selector")
theme = THEMES[selected_theme]
selected_page = st.sidebar.radio("Dashboard", ["Loan Risk Dashboard", "About Project"], index=0, key="page_selector")

FORM_DEFAULTS = {
    "age": 30,
    "annual_income": 60000,
    "credit_score": 650,
    "education": "High School",
    "experience": 5,
    "loan_amount": 50000,
    "loan_duration": 12,
    "dependents": 1,
    "monthly_debt": 1000,
    "utilization": 0.30,
    "open_credit": 5,
    "inquiries": 2,
    "dti": 0.30,
    "bankruptcy": "No",
    "previous_default": "No",
    "payment_history": 80,
    "credit_history": 10,
    "savings": 50000,
    "checking": 20000,
    "assets": 200000,
    "liabilities": 50000,
    "utility_history": 0.80,
    "job_tenure": 5,
    "employment": "Employed",
    "marital": "Divorced",
    "home": "Mortgage",
    "purpose": "Auto",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: __APP_BG__;
        color: __TEXT__;
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.2rem;
        max-width: 1240px;
    }

    [data-testid="stSidebar"] {
        background: __SIDEBAR_BG__;
    }

    [data-testid="stSidebar"] * {
        color: __SIDEBAR_TEXT__ !important;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="collapsedControl"] button,
    [data-testid="stHeader"] button[kind="header"] {
        color: #0f172a !important;
        background: rgba(255, 255, 255, 0.55) !important;
        border: 1px solid rgba(15, 23, 42, 0.18) !important;
        border-radius: 8px !important;
    }

    [data-testid="collapsedControl"] button svg,
    [data-testid="stHeader"] button[kind="header"] svg {
        fill: #0f172a !important;
        stroke: #0f172a !important;
    }

    .title {
        font-size: clamp(1.9rem, 3vw, 2.65rem);
        line-height: 1.12;
        font-weight: 800;
        color: __TEXT__;
        margin: 0;
        letter-spacing: 0;
    }

    .subtitle {
        color: __MUTED__;
        font-size: 1.03rem;
        margin-top: 0.7rem;
        max-width: 760px;
    }

    .hero-shell {
        margin-bottom: 1rem;
    }

    .hero-panel {
        background: __PANEL__;
        border: 1px solid __LINE__;
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 10px 30px __SHADOW__;
    }

    .hero-kpis {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 1.25rem;
    }

    .mini-kpi {
        border: 1px solid __LINE__;
        background: __PANEL__;
        border-radius: 8px;
        padding: 0.78rem 0.85rem;
    }

    .mini-kpi span {
        display: block;
        color: __MUTED__;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .mini-kpi strong {
        display: block;
        color: __TEXT__;
        font-size: 1.02rem;
        margin-top: 0.2rem;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 1.35rem 0 0.8rem;
    }

    .section-title .section-index {
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: __PANEL_SOFT__;
        color: __BRAND__;
        font-size: 1.1rem;
        font-weight: 800;
    }

    .section-title h2 {
        font-size: 1.25rem;
        margin: 0;
        color: __TEXT__;
        line-height: 1.2;
    }

    .section-title p {
        margin: 0.12rem 0 0;
        color: __MUTED__;
        font-size: 0.92rem;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    label,
    .stSlider label,
    .stSelectbox label,
    .stNumberInput label {
        color: __TEXT__ !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: __PANEL__ !important;
        color: __TEXT__ !important;
        border: 1px solid __LINE__ !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span,
    div[data-testid="stNumberInput"] input {
        color: __TEXT__ !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background-color: __BRAND__ !important;
        border-color: __BRAND__ !important;
    }

    .stButton > button,
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    .stButton > button:visited {
        background: #111111 !important;
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        min-height: 3.2rem !important;
        font-weight: 800 !important;
        box-shadow: 0 16px 34px __SHADOW__ !important;
        outline: none !important;
    }

    .stButton > button:hover {
        transform: none !important;
    }

    [data-testid="stAlert"] {
        background: __PANEL__ !important;
        color: __TEXT__ !important;
        border: 1px solid __LINE__ !important;
        border-radius: 8px !important;
    }

    [data-testid="stAlert"] * {
        color: __TEXT__ !important;
    }

    [data-testid="stMetric"] {
        background: __PANEL__;
        border: 1px solid __LINE__;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 28px __SHADOW__;
    }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-bottom: 1rem;
    }

    .result-card {
        background: __PANEL__;
        border: 1px solid __LINE__;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 28px __SHADOW__;
    }

    .result-card span {
        display: block;
        color: __MUTED__;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .result-card strong {
        display: block;
        color: __TEXT__;
        font-size: 1.45rem;
        margin-top: 0.2rem;
    }

    .result-card.approved {
        border-color: #22c55e;
        background: rgba(34, 197, 94, 0.16);
    }

    .result-card.warning {
        border-color: #f59e0b;
        background: rgba(245, 158, 11, 0.16);
    }

    .result-card.rejected {
        border-color: #ef4444;
        background: rgba(239, 68, 68, 0.16);
    }

    .reason-list {
        background: __PANEL__;
        border: 1px solid __LINE__;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.7rem 0;
        box-shadow: 0 10px 28px __SHADOW__;
    }

    .reason-item {
        display: flex;
        gap: 0.55rem;
        align-items: flex-start;
        padding: 0.45rem 0;
        color: __TEXT__;
        font-weight: 500;
    }

    .about-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 0.8rem;
    }

    .about-card {
        background: __PANEL__;
        border: 1px solid __LINE__;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 28px __SHADOW__;
    }

    .about-card h3 {
        margin: 0 0 0.45rem;
        color: __TEXT__;
        font-size: 1.02rem;
    }

    .about-card p {
        margin: 0;
        color: __MUTED__;
        font-size: 0.94rem;
        line-height: 1.5;
    }

    .footer {
        text-align: center;
        color: __MUTED__;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid __LINE__;
    }

    @media (max-width: 900px) {
        .hero-shell,
        .hero-kpis,
        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """.strip()
    .replace("__APP_BG__", theme["app_bg"])
    .replace("__APP_PATTERN__", theme["app_pattern"])
    .replace("__SIDEBAR_BG__", theme["sidebar_bg"])
    .replace("__SIDEBAR_TEXT__", theme["sidebar_text"])
    .replace("__PANEL__", theme["panel"])
    .replace("__PANEL_SOFT__", theme["panel_soft"])
    .replace("__TEXT__", theme["text"])
    .replace("__MUTED__", theme["muted"])
    .replace("__LINE__", theme["line"])
    .replace("__SHADOW__", theme["shadow"]),
    unsafe_allow_html=True,
)

def section_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <span class="section-index">{icon}</span>
            <div>
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_form_inputs() -> None:
    for key, value in FORM_DEFAULTS.items():
        st.session_state[key] = value


def render_about_dashboard() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-panel">
                <div class="title">About This Project</div>
                <div class="subtitle">AI-Based Loan Risk Assessment and Credit Decision Support System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("A", "Project Overview", "What this system does and why it is useful in credit decision workflows.")
    st.markdown(
        """
        <div class="about-card">
            <p>This dashboard predicts whether a loan application should be approved or rejected and provides interpretable supporting signals such as approval probability, risk level, reasons, and personalized recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("B", "Inputs & Outputs", "Key feature groups and decision outputs provided by the dashboard.")
    st.markdown(
        f"""
        <div class="about-grid">
            <div class="about-card">
                <h3>Input Coverage</h3>
                <p>{len(RAW_FEATURES)} applicant attributes spanning profile, credit history, liabilities, assets, and loan context.</p>
            </div>
            <div class="about-card">
                <h3>Decision Output</h3>
                <p>Final decision (Approved/Rejected), approval probability, and inferred risk level to support analyst review.</p>
            </div>
            <div class="about-card">
                <h3>Explainability Layer</h3>
                <p>Rule-based reason summaries and recommendation hints generated from applicant conditions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("C", "Technology Stack", "Implementation components used in this dashboard.")
    st.markdown(
        """
        <div class="about-grid">
            <div class="about-card"><h3>Frontend</h3><p>Streamlit with custom CSS-based theming and responsive section layouts.</p></div>
            <div class="about-card"><h3>ML/Data</h3><p>Python, pandas, NumPy, pickle model loading, and Matplotlib for probability visualization.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("D", "Workflow", "How to use the primary prediction dashboard.")
    st.markdown(
        """
        <div class="about-card">
            <p>1) Enter applicant profile details. 2) Add loan context attributes. 3) Click Predict Loan Approval. 4) Review decision, risk, chart, reasons, and recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-panel">
            <div class="title">AI-Based Loan Risk Assessment</div>
            <div class="subtitle">Credit decision support dashboard for reviewing applicant strength, approval probability, and risk signals in one focused workspace.</div>
            <div class="hero-kpis">
                <div class="mini-kpi"><span>Inputs</span><strong>{len(RAW_FEATURES)} applicant fields</strong></div>
                <div class="mini-kpi"><span>Output</span><strong>Decision + risk</strong></div>
                <div class="mini-kpi"><span>Review</span><strong>Applicant profile</strong></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Project Details")
    st.write("**Current Dashboard**")
    st.write(selected_page)
    st.markdown("---")
    st.write("**Project Title**")
    st.write("AI-Based Loan Risk Assessment and Credit Decision Support System")
    st.write("**Student Name**")
    st.write("Deekshitha R")
    st.markdown("---")
    st.write("Fill in the applicant details, then click the prediction button to get a complete decision summary.")

if selected_page == "About Project":
    render_about_dashboard()
    st.markdown('<div class="footer">Developed for AI-Based Loan Risk Assessment - Secure, explainable, and presentation-ready banking analytics</div>', unsafe_allow_html=True)
    st.stop()


section_header("1", "Applicant Profile", "Core personal, income, credit, and balance-sheet details used by the existing model.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 80, 30, key="age")
    annual_income = st.number_input("Annual Income", min_value=15000, max_value=500000, value=60000, key="annual_income")
    credit_score = st.slider("Credit Score", 300, 850, 650, key="credit_score")
    education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], key="education")
    experience = st.slider("Years of Experience", 0, 40, 5, key="experience")
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=200000, value=50000, key="loan_amount")
    loan_duration = st.selectbox("Loan Duration (Months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], key="loan_duration")
    dependents = st.slider("Number of Dependents", 0, 5, 1, key="dependents")
    monthly_debt = st.number_input("Monthly Debt Payments", min_value=0, max_value=30000, value=1000, key="monthly_debt")
    utilization = st.slider("Credit Card Utilization Rate", 0.0, 1.0, 0.30, key="utilization")
    open_credit = st.slider("Number of Open Credit Lines", 0, 20, 5, key="open_credit")

with col2:
    inquiries = st.slider("Number of Credit Inquiries", 0, 10, 2, key="inquiries")
    dti = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.30, key="dti")
    bankruptcy = st.selectbox("Bankruptcy History", ["No", "Yes"], key="bankruptcy")
    previous_default = st.selectbox("Previous Loan Default", ["No", "Yes"], key="previous_default")
    payment_history = st.slider("Payment History", 0, 100, 80, key="payment_history")
    credit_history = st.slider("Length of Credit History", 0, 40, 10, key="credit_history")
    savings = st.number_input("Savings Account Balance", min_value=0, max_value=1000000, value=50000, key="savings")
    checking = st.number_input("Checking Account Balance", min_value=0, max_value=500000, value=20000, key="checking")
    assets = st.number_input("Total Assets", min_value=0, max_value=10000000, value=200000, key="assets")
    liabilities = st.number_input("Total Liabilities", min_value=0, max_value=10000000, value=50000, key="liabilities")
    utility_history = st.slider("Utility Bills Payment History", 0.0, 1.0, 0.80, key="utility_history")
    job_tenure = st.slider("Job Tenure", 0, 20, 5, key="job_tenure")

section_header("2", "Loan Context", "Employment, household, ownership, and purpose attributes.")

context_col1, context_col2, context_col3, context_col4 = st.columns(4)
with context_col1:
    employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"], key="employment")
with context_col2:
    marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Widowed"], key="marital")
with context_col3:
    home = st.selectbox("Home Ownership", ["Mortgage", "Other", "Own", "Rent"], key="home")
with context_col4:
    purpose = st.selectbox("Loan Purpose", ["Auto", "Debt Consolidation", "Education", "Home", "Other"], key="purpose")

action_col1, action_col2 = st.columns(2)
with action_col1:
    predict = st.button("Predict Loan Approval", use_container_width=True)
with action_col2:
    st.button("Clear Inputs", use_container_width=True, on_click=clear_form_inputs)

if predict:
    bankruptcy_value = 1 if bankruptcy == "Yes" else 0
    default_value = 1 if previous_default == "Yes" else 0

    input_data = {
        "Age": age,
        "AnnualIncome": annual_income,
        "CreditScore": credit_score,
        "EducationLevel": education,
        "Experience": experience,
        "LoanAmount": loan_amount,
        "LoanDuration": loan_duration,
        "NumberOfDependents": dependents,
        "MonthlyDebtPayments": monthly_debt,
        "CreditCardUtilizationRate": utilization,
        "NumberOfOpenCreditLines": open_credit,
        "NumberOfCreditInquiries": inquiries,
        "DebtToIncomeRatio": dti,
        "BankruptcyHistory": bankruptcy_value,
        "PreviousLoanDefaults": default_value,
        "PaymentHistory": payment_history,
        "LengthOfCreditHistory": credit_history,
        "SavingsAccountBalance": savings,
        "CheckingAccountBalance": checking,
        "TotalAssets": assets,
        "TotalLiabilities": liabilities,
        "UtilityBillsPaymentHistory": utility_history,
        "JobTenure": job_tenure,
        "EmploymentStatus": employment,
        "MaritalStatus": marital,
        "HomeOwnershipStatus": home,
        "LoanPurpose": purpose,
    }

    encoded = encode_input(input_data)
    # Some saved models may return non-normalized class scores instead of
    # probabilities (due to versioning or custom wrappers). Normalize to ensure
    # we always present a valid probability in [0,1].
    proba_raw = model.predict_proba(encoded)[0]
    try:
        proba_arr = np.asarray(proba_raw, dtype=float)
        total = np.nansum(proba_arr)
        if total <= 0 or not np.isfinite(total):
            probability = 0.0
        else:
            proba_norm = proba_arr / total
            probability = float(proba_norm[1]) if proba_norm.shape[0] > 1 else float(proba_norm[0])
    except Exception:
        # Last-resort fallback: attempt to coerce the second element
        try:
            probability = float(proba_raw[1])
        except Exception:
            probability = 0.0
    prediction = int(model.predict(encoded)[0])

    if prediction == 1:
        decision = "Loan Approved"
        if probability >= 0.60:
            risk_level = "Low"
            status_color = "success"
        else:
            risk_level = "Medium"
            status_color = "warning"
    else:
        decision = "Loan Rejected"
        risk_level = "High"
        status_color = "error"

    section_header("3", "Credit Decision Outcome", "Prediction output from the trained model with supporting risk notes.")

    result_class = "approved" if status_color == "success" else "warning" if status_color == "warning" else "rejected"
    st.markdown(
        f"""
        <div class="result-grid">
            <div class="result-card {result_class}"><span>Decision</span><strong>{decision}</strong></div>
            <div class="result-card"><span>Approval Probability</span><strong>{probability * 100:.1f}%</strong></div>
            <div class="result-card"><span>Risk Level</span><strong>{risk_level}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    

    approved = probability * 100
    rejected = 100 - approved

    fig, ax = plt.subplots(figsize=(2.7, 2.7), facecolor=theme["panel"])
    ax.set_facecolor(theme["panel"])

    ax.pie(
        [approved, rejected],
        labels=["Approved", "Rejected"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.08, 0),
        shadow=False,
        colors=["#14b8a6", "#f59e0b"],
        wedgeprops={"linewidth": 2, "edgecolor": theme["panel"]},
        textprops={"color": theme["text"], "fontsize": 9, "fontweight": "bold"},
    )     

    ax.set_title("Loan Approval Probability", fontsize=11, fontweight="bold", color=theme["text"])
    st.pyplot(fig, use_container_width=False)   
    if status_color == "success":
        st.success("Strong applicant profile. The model indicates a low-risk borrowing profile.")
    elif status_color == "warning":
        st.warning("Moderate risk detected. Further review is recommended before final approval.")
    else:
        st.error("High-risk profile. The model suggests the application should be rejected or escalated.")

    st.markdown("### Credit Decision Support")

    if decision == "Loan Approved":
        st.success("The applicant demonstrates a balanced financial profile and appears capable of meeting repayment obligations.")

        st.markdown("### Why Approved?")

        reasons = []

        if credit_score >= 750:
            reasons.append("Excellent Credit Score")

        if bankruptcy_value == 0:
            reasons.append("No Bankruptcy History")

        if default_value == 0:
            reasons.append("No Previous Loan Defaults")

        if dti <= 0.35:
            reasons.append("Low Debt-to-Income Ratio")

        if annual_income >= 70000:
            reasons.append("Stable Annual Income")

        if payment_history >= 75:
            reasons.append("Good Payment History")

        if assets >= 100000:
            reasons.append("Strong Asset Holdings")

        reason_html = "".join([f'<div class="reason-item">OK <span>{reason}</span></div>' for reason in reasons])
        st.markdown(f'<div class="reason-list">{reason_html}</div>', unsafe_allow_html=True)

    else:
        st.error("The applicant shows elevated credit risk due to weaker financial indicators and should be reviewed carefully.")

        st.markdown("### Why Rejected?")

        reasons = []

        if credit_score < 650:
            reasons.append("Low Credit Score")

        if bankruptcy_value == 1:
            reasons.append("Bankruptcy History Found")

        if default_value == 1:
            reasons.append("Previous Loan Default")

        if dti > 0.50:
            reasons.append("High Debt-to-Income Ratio")

        if annual_income < 70000:
            reasons.append("Low Annual Income")

        if payment_history < 75:
            reasons.append("Poor Payment History")

        if assets < 100000:
            reasons.append("Low Asset Holdings")

        reason_html = "".join([f'<div class="reason-item">Review <span>{reason}</span></div>' for reason in reasons])
        st.markdown(f'<div class="reason-list">{reason_html}</div>', unsafe_allow_html=True)
    st.markdown("### Personalized Recommendations")
    recommendations = []
    if annual_income < 70000:
        recommendations.append("Increase documented income stability or add a guarantor to strengthen repayment capacity.")
    if loan_amount > 80000:
        recommendations.append("Reduce the requested loan amount to better align with the applicant's financial capacity.")
    if assets < 100000:
        recommendations.append("Build stronger savings or collateral support to improve underwriting confidence.")
    if loan_duration > 60:
        recommendations.append("Consider a shorter repayment term to reduce long-term repayment burden.")
    if payment_history < 75:
        recommendations.append("Improve payment consistency and resolve past dues to strengthen credit reliability.")
    if default_value == 1:
        recommendations.append("Address prior loan default issues before reapplying for new credit.")
    if bankruptcy_value == 1:
        recommendations.append("Work on restoring credit health and providing a stronger financial explanation for the application.")
    if not recommendations:
        recommendations.append("Maintain current financial discipline and continue monitoring repayment behavior.")

    recommendation_html = "".join([f'<div class="reason-item">Tip <span>{item}</span></div>' for item in recommendations])
    st.markdown(f'<div class="reason-list">{recommendation_html}</div>', unsafe_allow_html=True)
    st.components.v1.html(f"""
<script>
function printReport(){{
    var w = window.open('', '', 'height=700,width=900');
    w.document.write(`
        <h1>AI-Based Loan Risk Assessment Report</h1>

        <hr>

        <h3>Applicant Details</h3>

        <p><b>Age:</b> {age}</p>
        <p><b>Income:</b> {annual_income}</p>
        <p><b>Credit Score:</b> {credit_score}</p>
        <p><b>Education:</b> {education}</p>
        <p><b>Employment:</b> {employment}</p>

        <hr>

        <h3>Prediction</h3>

        <p><b>Decision:</b> {decision}</p>
        <p><b>Approval Probability:</b> {(probability*100):.2f}%</p>
        <p><b>Risk Level:</b> {risk_level}</p>
    `);

    w.document.close();
    w.print();
}}

</script>

<button onclick="printReport()"
style="
background:#1976D2;
color:white;
padding:15px;
font-size:18px;
border:none;
border-radius:8px;">
🖨️ Download Report
</button>
""", height=80)

    st.markdown("---")
    st.caption("Prediction generated using the trained loan risk assessment system and the engineered feature set.")

st.markdown('<div class="footer">Developed for AI-Based Loan Risk Assessment - Secure, explainable, and presentation-ready banking analytics</div>', unsafe_allow_html=True)

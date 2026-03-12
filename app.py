import joblib, json
import streamlit as st
import numpy as np
import pandas as pd
import requests
import re
from urllib.parse import urlparse
import pdfplumber
from PIL import Image
import pytesseract

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_pipeline(folder):
    pipe = joblib.load(f"{folder}/pipeline.pkl")
    with open(f"{folder}/meta.json", "r") as f:
        meta = json.load(f)
    return pipe, meta

def make_X(meta, data_dict):
    cols = meta["feature_cols"]
    return pd.DataFrame([{c: data_dict.get(c, np.nan) for c in cols}])

# Load both models at start
eng_pipe, eng_meta = load_pipeline("artifacts_engineering")
bus_pipe, bus_meta = load_pipeline("artifacts_business")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CareerPulse", page_icon="✨", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1100px; }
h1, h2, h3 { letter-spacing: 0.2px; }

.small-muted { color: #9CA3AF; font-size: 0.92rem; margin-top: -8px; }

.hr {
  border: 0; height: 1px; background: #1F2937;
  margin: 18px 0 22px 0;
}

.card {
  background: #0F172A;               /* deep navy */
  border: 1px solid #1F2937;
  border-radius: 18px;
  padding: 18px 18px;
  margin-bottom: 18px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.18);
}

.card-soft {
  background: #0B1220;
  border: 1px dashed #243044;
  border-radius: 18px;
  padding: 16px 18px;
  margin-bottom: 18px;
  opacity: 0.92;
}

.badge {
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  background:#7C3AED;
  color:white;
  font-size:12px;
  margin-bottom: 8px;
}

.metric {
  font-size: 2.1rem;
  font-weight: 700;
  margin: 0;
}

.metric-sub {
  color: #9CA3AF;
  margin-top: 4px;
  font-size: 0.95rem;
}

.pill {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(124,58,237,0.18);
  border: 1px solid rgba(124,58,237,0.35);
  color: #E9D5FF;
  font-size: 0.85rem;
  margin-right: 8px;
  margin-bottom: 8px;
}

.btn-row { margin-top: 10px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("✨ CareerPulse")
st.markdown(
    "<div class='small-muted'>Placement prediction and employability readiness from academics, CV, and portfolio.</div>",
    unsafe_allow_html=True
)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def clean_ocr_text(text: str) -> str:
    text = text.lower()

    replacements = {
        "pueance": "python",
        "jave": "java",
        "exel": "excel",
        "ms exel": "excel",
        "basicmachine learning": "machine learning",
        "datastructurs": "data structures",
        "powerpoint cece": "powerpoint",
        "operating systems fundamentals": "operating systems",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_skills(text_l: str) -> list:
    skill_keywords = {
        "Python": [r"\bpython\b", r"p\s*y\s*t\s*h\s*o\s*n"],
        "Java": [r"\bjava\b", r"j\s*a\s*v\s*a"],
        "C++": [r"c\+\+"],
        "C": [r"\bc\b"],
        "SQL": [r"\bsql\b", r"m\s*y\s*s\s*q\s*l"],
        "Machine Learning": [r"machine learning", r"\bml\b"],
        "Deep Learning": [r"deep learning", r"\bdl\b"],
        "Data Structures": [r"data structures", r"\bdsa\b"],
        "Operating Systems": [r"operating systems", r"\bos\b"],
        "HTML": [r"\bhtml\b"],
        "CSS": [r"\bcss\b"],
        "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
        "React": [r"\breact\b"],
        "Git/GitHub": [r"\bgit\b", r"github"],
        "Excel": [r"\bexcel\b"],
        "PowerPoint": [r"\bpowerpoint\b"],
        "MS Word": [r"\bword\b", r"ms word"],
    }

    found = []
    for skill, patterns in skill_keywords.items():
        if any(re.search(p, text_l) for p in patterns):
            found.append(skill)
    return sorted(list(set(found)))

def extract_text_from_cv(cv_file) -> str:
    text = ""
    file_type = cv_file.type

    if file_type == "application/pdf":
        try:
            with pdfplumber.open(cv_file) as pdf:
                for page in pdf.pages[:5]:
                    text += (page.extract_text() or "") + "\n"
        except Exception:
            text = ""

    elif file_type.startswith("image"):
        try:
            image = Image.open(cv_file)
            text = pytesseract.image_to_string(image)
        except Exception:
            text = ""

    return text.strip()

# ---------------- BASIC INPUTS ----------------
st.subheader("Profile")

stream = st.selectbox(
    "Choose Stream Category",
    ["Engineering", "Business & Management", "Other (Readiness Only)"]
)

role = st.selectbox(
    "Target Role",
    ["Software Developer", "Data Analyst", "ML/AI", "Web Developer",
     "Business/Marketing", "HR", "Finance", "Other"]
)

col1, col2 = st.columns(2)
with col1:
    cgpa = st.number_input("CGPA (0–10)", 0.0, 10.0, 7.0, 0.1)
with col2:
    backlogs = st.number_input("Backlogs", 0, 20, 0)

# -------- DEFAULT VALUES (prevent NameError) --------
age = 21
gender_eng = "Male"
stream_eng = "Computer Science"
hostel = 0
history_backlogs = 0
internships_eng = 0

gender_bus = "M"
specialisation = "Mkt&Fin"
ssc_p = hsc_p = degree_p = etest_p = mba_p = 70.0
workex = "No"

# ---------------- STREAM-SPECIFIC INPUTS ----------------
if stream == "Engineering":
    st.subheader("Engineering Placement Inputs")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 16, 40, 21)
    with c2:
        gender_eng = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        stream_eng = st.selectbox("Engineering Stream", [
            "Computer Science", "Information Technology",
            "Electronics And Communication", "Mechanical", "Civil", "Electrical"
        ])
    with c4:
        hostel = st.selectbox("Hostel (0=No, 1=Yes)", [0, 1])

    history_backlogs = st.selectbox("History Of Backlogs (0/1)", [0, 1])
    internships_eng = st.number_input("Internships (count)", 0, 20, 0)

elif stream == "Business & Management":
    st.subheader("Business Placement Inputs (Campus Recruitment)")

    c1, c2, c3 = st.columns(3)
    with c1:
        ssc_p = st.number_input("SSC %", 0.0, 100.0, 70.0, 0.1)
        hsc_p = st.number_input("HSC %", 0.0, 100.0, 70.0, 0.1)
    with c2:
        degree_p = st.number_input("Degree %", 0.0, 100.0, 65.0, 0.1)
        etest_p = st.number_input("E-test %", 0.0, 100.0, 70.0, 0.1)
    with c3:
        mba_p = st.number_input("MBA %", 0.0, 100.0, 65.0, 0.1)
        workex = st.selectbox("Work Experience", ["Yes", "No"])

    gender_bus = st.selectbox("Gender (M/F)", ["M", "F"])
    specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])

# ---------------- CV UPLOAD ----------------
st.subheader("Upload CV")

cv_file = st.file_uploader(
    "Upload your CV (PDF or Image)",
    type=["pdf", "png", "jpg", "jpeg"]
)

skills_found = []
internships = 0
projects = 0
certs = 0
companies = []
raw_text = ""
text_l = ""

if cv_file:
    raw_text = extract_text_from_cv(cv_file)

    if len(raw_text.strip()) < 30 and cv_file.type == "application/pdf":
        st.warning("This PDF looks like an image-based PDF. Upload it as JPG/PNG for OCR.")
    else:
        text_l = clean_ocr_text(raw_text)

        # ✅ actually detect skills
        skills_found = detect_skills(text_l)

        # Better keyword counts
        internships = len(re.findall(r"\bintern\b|\binternship\b|\btraining\b|\btrainee\b", text_l))
        projects = len(re.findall(r"\bproject\b|\bprojects\b|\bcapstone\b", text_l))
        certs = len(re.findall(r"\bcertification\b|\bcertificate\b|\bcredential\b|\bcourse\b", text_l))

        company_matches = re.findall(
            r"(intern|worked)\s+(at|with)\s+([A-Z][A-Za-z0-9 &]+)",
            raw_text
        )
        companies = sorted(list(set([m[2] for m in company_matches])))

        st.markdown("### 🛠️ Skills Detected")
        if skills_found:
            st.success(", ".join(skills_found))
        else:
            st.warning("No skills detected. Try uploading a clearer resume image or add a 'Skills' section in text.")

        st.markdown("### 🔎 Extracted Text Preview (first 900 chars)")
        st.text_area("Preview", raw_text[:900], height=200)

# ---------------- GITHUB ----------------
st.subheader("GitHub (Optional)")
github_url = st.text_input("GitHub profile URL (example: https://github.com/username)")

gh_repos = 0
gh_stars = 0

if github_url:
    try:
        username = urlparse(github_url).path.strip("/")
        user = requests.get(f"https://api.github.com/users/{username}", timeout=10).json()
        repos = requests.get(user["repos_url"], timeout=10).json()

        gh_repos = len(repos) if isinstance(repos, list) else 0
        if isinstance(repos, list):
            gh_stars = sum(r.get("stargazers_count", 0) for r in repos)
    except Exception:
        st.warning("Could not fetch GitHub data. Check the URL or your internet connection.")

# ---------------- ANALYZE ----------------
if st.button("Analyze"):


    st.info(
    "ℹ️ **Important Note:** Placement Prediction and Employability Readiness are "
    "calculated using **different criteria**. "
    "Placement Prediction is based on historical campus placement data "
    "(academics and past trends), while Employability Readiness reflects "
    "your current skills, projects, internships, and portfolio strength. "
    "It is possible to have a high placement probability but a lower readiness score."
    )

    st.markdown("## Results")

    # -------- Placement Prediction --------
    if stream == "Engineering":
        provided = {
            "Age": age,
            "Gender": gender_eng,
            "Stream": stream_eng,
            "Internships": internships_eng,
            "CGPA": cgpa,
            "Hostel": hostel,
            "HistoryOfBacklogs": history_backlogs,
        }
        X = make_X(eng_meta, provided)
        prob = float(eng_pipe.predict_proba(X)[0][1])
        pred = "Placed" if prob >= 0.5 else "Not Placed"

        st.markdown(f"""
        <div class="card">
          <div class="badge">Engineering Placement Prediction</div>
          <h2>{pred}</h2>
          <p>Probability: <b>{prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    elif stream == "Business & Management":
        provided = {
            "ssc_p": ssc_p,
            "hsc_p": hsc_p,
            "degree_p": degree_p,
            "etest_p": etest_p,
            "mba_p": mba_p,
            "workex": workex,
            "gender": gender_bus,
            "specialisation": specialisation,
        }
        X = make_X(bus_meta, provided)
        prob = float(bus_pipe.predict_proba(X)[0][1])
        pred = "Placed" if prob >= 0.5 else "Not Placed"

        st.markdown(f"""
        <div class="card">
          <div class="badge">Business Placement Prediction</div>
          <h2>{pred}</h2>
          <p>Probability: <b>{prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Other streams: Placement prediction not available. Showing readiness score only.")

    # -------- Employability Readiness Score --------
    score = 0.0

    # Academics (max 40)
    score += (cgpa / 10.0) * 40.0
    score -= backlogs * 3.0

    # Resume signals (max ~61)
    score += min(len(skills_found) * 5.0, 25.0)
    score += min(internships * 6.0, 18.0)
    score += min(projects * 4.0, 12.0)
    score += min(certs * 2.0, 6.0)

    # GitHub (max 15)
    score += min(gh_repos * 2.0, 10.0)
    score += min(gh_stars * 0.5, 5.0)

    score = max(0.0, min(score, 100.0))

    st.markdown(f"""
    <div class="card">
      <div class="badge">Employability Readiness</div>
      <h2>{score:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    if score < 60:
        st.warning("Low readiness detected. Improve skills, projects, and internships.")
    else:
        st.success("Good readiness level!")

    # -------- Details --------
    st.subheader("Detected Information")
    st.write("**Skills:**", skills_found or "None detected")
    st.write("**Internships (from CV):**", internships)
    st.write("**Projects:**", projects)
    st.write("**Certifications:**", certs)
    st.write("**Companies:**", companies or "None")
    st.write("**GitHub Repos:**", gh_repos, "| **Stars:**", gh_stars)

    # -------- Suggestions --------
    st.subheader("📌 Suggestions to Improve Employability")
    suggestions = []

    if "Python" not in skills_found:
        suggestions.append("Learn Python (high demand across roles).")
    if stream == "Engineering" and "Machine Learning" not in skills_found:
        suggestions.append("Explore basic ML concepts + 1 mini project.")
    if internships == 0:
        suggestions.append("Try to secure at least one internship (even remote).")
    if projects < 2:
        suggestions.append("Build 2+ projects and upload them to GitHub with READMEs.")
    if gh_repos == 0 and github_url == "":
        suggestions.append("Create a GitHub profile and upload your projects.")
    if backlogs > 0:
        suggestions.append("Clear backlogs; they reduce placement chances in many companies.")

    if not suggestions:
        suggestions.append("Your profile looks strong. Keep applying and improving consistently!")

    for s in suggestions:
        st.write("•", s)



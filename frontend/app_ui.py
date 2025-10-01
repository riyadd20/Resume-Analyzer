import streamlit as st
import requests
import fitz

# API Endpoint
BASE_URL = "https://resume-analyzer-xaf6.onrender.com"

def safe_api_error(response):
    try:
        return response.json()
    except Exception:
        return response.text

def extract_text_from_pdf(uploaded_file):
    """Extracts text from uploaded PDF resume."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text.strip()

st.title("📄 AI-Powered Job Analyzer")

# Job Description
st.subheader("🔍 Job Description")
job_desc = st.text_area("Paste the job description here...", height=150)

# Resume Input
st.subheader("💼 Resume")
resume_option = st.radio("Choose resume input method:", ["Paste text", "Upload PDF"])

resume_text = ""
if resume_option == "Paste text":
    resume_text = st.text_area("Paste your resume here...", height=150)
elif resume_option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)

st.markdown("<hr>", unsafe_allow_html=True)

# Buttons
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Results
result_placeholder = st.container()

# Buttons
with col1:
    if st.button("Analyze Match Score", use_container_width=True):
        if job_desc.strip() == "" or resume_text.strip() == "":
            result_placeholder.error("⚠️ Please enter both Job Description and Resume!")
        else:
            with result_placeholder:
                with st.spinner("Calculating match score... ⏳"):
                    response = requests.post(f"{BASE_URL}/match_score",
                                             json={"job_desc": job_desc, "resume_text": resume_text})
                if response.status_code == 200:
                    scores = response.json()["scores"]
                    st.subheader("Resume Match Scores:")

                    # Show details
                    st.write(f"- MiniLM (cosine): {scores['MiniLM']}")
                    st.write(f"- MPNet (cosine): {scores['MPNet']}")
                    st.write(f"- Cosine Similarity (avg): {scores['Cosine Similarity']}")

                    # Final normalized score
                    st.markdown(f"**Final Score: {scores['Final Score']}**")

                    # Category with color
                    category = scores["Category"]
                    if category == "Strong Match":
                        st.success(f"🟢 {category}")
                    elif category == "Moderate Match":
                        st.info(f"🔵 {category}")
                    elif category == "Partial Match":
                        st.warning(f"🟠 {category}")
                    else:
                        st.error(f"🔴 {category}")
                else:
                    st.error(f"⚠️ API Error: {safe_api_error(response)}")

with col2:
    if st.button("Get AI Resume Feedback", use_container_width=True):
        if job_desc.strip() == "" or resume_text.strip() == "":
            result_placeholder.error("⚠️ Please enter both Job Description and Resume!")
        else:
            with result_placeholder:
                with st.spinner("Analyzing resume... ⏳"):
                    response = requests.post(f"{BASE_URL}/resume_feedback",
                                             json={"job_desc": job_desc, "resume_text": resume_text})
                if response.status_code == 200:
                    st.subheader("📋 AI Resume Feedback:")
                    st.write(response.json().get("feedback", "No feedback returned."))
                else:
                    st.error(f"⚠️ API Error: {safe_api_error(response)}")

with col3:
    if st.button("Generate AI Resume", use_container_width=True):
        if job_desc.strip() == "":
            result_placeholder.error("⚠️ Please enter a job description!")
        else:
            with result_placeholder:
                with st.spinner("Generating resume... ⏳"):
                    response = requests.post(f"{BASE_URL}/generate_resume",
                                             json={"job_desc": job_desc})
                if response.status_code == 200:
                    st.subheader("📄 AI-Generated Resume:")
                    st.write(response.json().get("resume", "No resume generated."))
                else:
                    st.error(f"⚠️ API Error: {safe_api_error(response)}")

with col4:
    if st.button("Get AI Interview Questions", use_container_width=True):
        if job_desc.strip() == "":
            result_placeholder.error("⚠️ Please enter a job description!")
        else:
            with result_placeholder:
                with st.spinner("Generating interview questions... ⏳"):
                    response = requests.post(f"{BASE_URL}/interview_questions",
                                             json={"job_desc": job_desc})
                if response.status_code == 200:
                    st.subheader("🎤 AI Interview Questions:")
                    st.write(response.json().get("questions", "No questions generated."))
                else:
                    st.error(f"⚠️ API Error: {safe_api_error(response)}")

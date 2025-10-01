from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from bert_model import get_resume_match_scores

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = FastAPI()

# Request Model
class JobRequest(BaseModel):
    job_desc: str
    resume_text: str = None

# Resume Match Score (Multi-model BERT)
@app.post("/match_score")
def match_score(request: JobRequest):
    try:
        scores = get_resume_match_scores(request.job_desc, request.resume_text)
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Match score error: {str(e)}")

# AI Resume Feedback (Gemini)
@app.post("/resume_feedback")
def resume_feedback(request: JobRequest):
    try:
        prompt = f"""
        Job Description: {request.job_desc}
        Resume: {request.resume_text}

        1. Provide a detailed analysis of the resume based on the job description.
        2. Rate the resume out of 100.
        3. Suggest missing skills grouped into categories (Core Programming, Frameworks, Cloud, etc.).
        4. Recommend 1-2 resources for each missing skill (like YouTube or Coursera).
        """
        response = genai.GenerativeModel("gemini-2.5-pro").generate_content(prompt)
        return {"feedback": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume feedback error: {str(e)}")

# AI Resume Generator
@app.post("/generate_resume")
def generate_resume(request: JobRequest):
    try:
        prompt = f"""
        You are a professional resume writer. Based on this job description:

        {request.job_desc}

        Generate a high-quality professional resume including:
        - A well-structured summary
        - Relevant work experience
        - Key technical and soft skills
        - Educational background
        - Any relevant projects
        """
        response = genai.GenerativeModel("gemini-2.5-pro").generate_content(prompt)
        return {"resume": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume generator error: {str(e)}")

# AI Interview Questions
@app.post("/interview_questions")
def interview_questions(request: JobRequest):
    try:
        prompt = f"""
        Based on the job description below:

        {request.job_desc}

        Generate:
        - 5 technical interview questions related to the required skills.
        - 5 behavioral interview questions to assess soft skills.
        """
        response = genai.GenerativeModel("gemini-2.5-pro").generate_content(prompt)
        return {"questions": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview questions error: {str(e)}")

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

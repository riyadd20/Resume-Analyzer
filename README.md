# Resume Analyzer

This project is a simple web app that compares a resume with a job description. It gives a match score, provides feedback, and can also generate a resume or interview questions based on the job description.

It’s built with a FastAPI backend and a Streamlit frontend.

---

## What it does

* Paste a job description
* Paste a resume or upload a PDF
* Get a match score between the resume and job
* Get feedback on what’s missing or can be improved
* Generate a sample resume from a job description
* Generate interview questions

---

## How it works

* The backend computes similarity between the resume and job description using sentence embeddings
* The frontend sends requests and displays results in a simple UI
* Gemini is used for feedback, resume generation, and interview questions

---

## Tech Stack

* Python
* FastAPI
* Streamlit
* Hugging Face embeddings
* Google Gemini
* PyMuPDF

---

## Project Structure

```text
Resume-Analyzer/
├── backend/
│   ├── app.py
│   ├── bert_model.py
│   └── requirements.txt
├── frontend/
│   ├── app_ui.py
│   └── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

### 2. Set up backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Set up frontend

```bash
cd ../frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file inside the backend folder:

```env
GEMINI_API_KEY=your_api_key
HF_API_TOKEN=your_huggingface_token
```

---

## Run the backend

```bash
cd backend
python app.py
```

Backend runs at:

```
http://localhost:8000
```

---

## Run the frontend

Before running, open `frontend/app_ui.py` and update:

```python
BASE_URL = "https://resume-analyzer-xaf6.onrender.com"
```

to:

```python
BASE_URL = "http://localhost:8000"
```

Then run:

```bash
cd frontend
streamlit run app_ui.py
```

---

## Using the app

1. Enter a job description
2. Paste a resume or upload a PDF
3. Choose what you want:

   * Match score
   * Resume feedback
   * Resume generation
   * Interview questions

---

## Notes

* Works best with text-based PDFs
* Scanned resumes may not extract well
* Match score is based on semantic similarity, not exact ATS logic

---

## Future Improvements

* Add OCR support for scanned PDFs
* Improve scoring logic
* Add history or saved results

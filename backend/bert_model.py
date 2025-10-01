import os
import requests
import numpy as np
import math
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
print("HF_API_TOKEN loaded:", HF_API_TOKEN[:6] + "..." if HF_API_TOKEN else "NOT FOUND")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

HF_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ---------- Helper functions ----------
def normalize_score(score):
    """
    Boost cosine similarity using logistic scaling.
    This makes 0.3–0.4 map to 60–80 realistically.
    """
    boosted = 1 / (1 + math.exp(-12 * (score - 0.35)))
    return round(boosted * 100, 2)

def categorize_score(score):
    """Categorize normalized score into match levels."""
    if score >= 80:
        return "Strong Match"
    elif score >= 60:
        return "Moderate Match"
    elif score >= 40:
        return "Partial Match"
    else:
        return "Weak Match"

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def get_hf_embedding(model_name, text):
    """Get embeddings from Hugging Face Inference API."""
    API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
    response = requests.post(API_URL, headers=HF_HEADERS, json={"inputs": text})
    response.raise_for_status()
    return np.array(response.json()[0])

# ---------- Main function ----------
def get_resume_match_scores(resume_text, job_text):
    # Encode texts using Hugging Face API
    resume_emb_minilm = get_hf_embedding(HF_MINILM_MODEL, resume_text)
    job_emb_minilm = get_hf_embedding(HF_MINILM_MODEL, job_text)
    resume_emb_mpnet = get_hf_embedding(HF_MPNET_MODEL, resume_text)
    job_emb_mpnet = get_hf_embedding(HF_MPNET_MODEL, job_text)

    # Cosine similarities
    minilm_score = cosine_similarity(resume_emb_minilm, job_emb_minilm)
    mpnet_score = cosine_similarity(resume_emb_mpnet, job_emb_mpnet)

    # Average
    final_cosine = (minilm_score + mpnet_score) / 2.0

    # Normalize into 0–100
    final_score = normalize_score(final_cosine)

    # Category
    category = categorize_score(final_score)

    return {
        "MiniLM": f"{minilm_score:.2f}",
        "MPNet": f"{mpnet_score:.2f}",
        "Final Score": f"{final_score:.0f}/100",
        "Cosine Similarity": f"{final_cosine:.2f}",
        "Category": category
    }

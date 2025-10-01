from sentence_transformers import SentenceTransformer, util
import math

# Load models
minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def normalize_score(score):
    """
    Boost cosine similarity using logistic scaling.
    This makes 0.3–0.4 map to 60–80 realistically.
    """
    # Logistic transformation
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

def get_resume_match_scores(resume_text, job_text):
    # Encode texts
    resume_emb_minilm = minilm_model.encode(resume_text, convert_to_tensor=True)
    job_emb_minilm = minilm_model.encode(job_text, convert_to_tensor=True)
    resume_emb_mpnet = mpnet_model.encode(resume_text, convert_to_tensor=True)
    job_emb_mpnet = mpnet_model.encode(job_text, convert_to_tensor=True)

    # Cosine similarities
    minilm_score = util.cos_sim(resume_emb_minilm, job_emb_minilm).item()
    mpnet_score = util.cos_sim(resume_emb_mpnet, job_emb_mpnet).item()

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

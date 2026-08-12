from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pipeline.pipeline import analyze_reviews
from src.pipeline.dashboard_service import build_dashboard

app = FastAPI()

# =========================
# CORS FIX
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

# =========================
# REQUEST MODEL
# =========================
class ReviewRequest(BaseModel):
    reviews: list[str]

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Review NLP API running"}

# =========================
# MAIN ENDPOINT
# =========================
@app.post("/analyze")
def analyze_reviews_api(request: ReviewRequest):
    results = analyze_reviews(request.reviews)
    dashboard = build_dashboard(results)

    return {
        "status": "success",
        "data": dashboard
    }
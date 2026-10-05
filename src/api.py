from fastapi import FastAPI, HTTPException, Query
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import recommend


app = FastAPI(
    title="EduTrack LMS Recommendation API",
    description="Adaptive learning path recommendation API using hybrid recommendation.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "project": "EduTrack LMS",
        "service": "Adaptive Learning Path Recommendation API",
        "status": "running"
    }


@app.get("/health")
def health():
    try:
        from src.recommender import load_model, load_datasets

        load_model()
        load_datasets()

        return {
            "status": "healthy",
            "model": "loaded",
            "service": "recommendation-api"
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Recommendation service unavailable: {str(e)}"
        )


@app.get("/recommendations/{student_id}")
def get_recommendations(
    student_id: str,
    top_n: int = Query(
        default=5,
        ge=1,
        le=10,
        description="Number of recommendations to return."
    )
):
    try:
        result = recommend(
            student_id=student_id,
            top_n=top_n
        )

        if not result["recommendations"]:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations available for student '{student_id}'."
            )

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation error: {str(e)}"
        )


@app.get("/students/{student_id}/recommendations")
def get_student_recommendations(
    student_id: str,
    top_n: int = Query(
        default=5,
        ge=1,
        le=10,
        description="Number of recommendations to return."
    )
):
    return get_recommendations(
        student_id=student_id,
        top_n=top_n
    )



from fastapi import FastAPI

app = FastAPI(
    title="SciFact Retrieval System",
    description="A reproducible research system for document retrieval.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""
    return {"status": "ok"}


import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.retrieval import BM25, DocumentRecord

load_dotenv()

database_url = os.getenv("DATABASE_URL")
engine = (
    create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    if database_url
    else None
)

app = FastAPI(
    title="SciFact Retrieval System",
    description="A reproducible research system for document retrieval.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class StatsResponse(BaseModel):
    documents: int
    chunks: int


def database_session():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured. Start the local database first.",
        )
    with Session(engine) as session:
        yield session


DatabaseSession = Annotated[Session, Depends(database_session)]


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse, tags=["Dataset"])
def dataset_stats(session: DatabaseSession) -> StatsResponse:
    """Return counts for the ingested corpus and generated chunks."""
    try:
        documents = session.scalar(select(func.count()).select_from(Document)) or 0
        chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503,
            detail="The local database is unavailable.",
        ) from error
    return StatsResponse(documents=documents, chunks=chunks)


@app.get("/search", response_model=SearchResponse, tags=["Retrieval"])
def search_documents(
    session: DatabaseSession,
    query: Annotated[str, Query(min_length=1, max_length=500)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SearchResponse:
    """Search the ingested corpus with the deterministic BM25 baseline."""
    try:
        documents = list(session.scalars(select(Document).order_by(Document.doc_id)))
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503,
            detail="The local database is unavailable.",
        ) from error
    if not documents:
        raise HTTPException(
            status_code=503,
            detail="No documents are ingested. Run the ingestion script first.",
        )

    records = [
        DocumentRecord(doc_id=doc.doc_id, title=doc.title, text=doc.text)
        for doc in documents
    ]
    index = BM25(records)
    scores = dict(index.search(query, limit))
    result_by_id = {record.doc_id: record for record in records}
    results = [
        SearchResult(
            doc_id=doc_id,
            title=result_by_id[doc_id].title,
            text=result_by_id[doc_id].text,
            score=score,
        )
        for doc_id, score in scores.items()
    ]
    return SearchResponse(query=query, results=results)

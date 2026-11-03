
import os
import logging
from pathlib import Path
import sys
from threading import Lock
import time
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.retrieval import BM25, DocumentRecord
from app.vector_retrieval import VectorRetriever
from app.hybrid_retrieval import HybridRetriever

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
logger = logging.getLogger(__name__)
_vector_embedder = None
_vector_embedder_lock = Lock()
_bm25_retriever = None
_bm25_retriever_lock = Lock()
_preload_complete = False
_preload_error: str | None = None
PRELOAD_ENV = "SCIFACT_PRELOAD"

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
    mode: str
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


def _get_vector_embedder(model: str):
    global _vector_embedder
    if _vector_embedder is None:
        with _vector_embedder_lock:
            if _vector_embedder is None:
                from scripts.eval.run_eval import SentenceTransformerQueryEmbedder

                started = time.perf_counter()
                _vector_embedder = SentenceTransformerQueryEmbedder(model)
                logger.info(
                    "Loaded vector model %s in %.3f seconds",
                    model,
                    time.perf_counter() - started,
                )
    return _vector_embedder


def _preload_enabled() -> bool:
    return os.getenv(PRELOAD_ENV, "").lower() in {"1", "true", "yes", "on"}


def _get_bm25_retriever(session: Session) -> BM25:
    global _bm25_retriever
    if _bm25_retriever is None:
        with _bm25_retriever_lock:
            if _bm25_retriever is None:
                documents = list(
                    session.scalars(select(Document).order_by(Document.doc_id))
                )
                if not documents:
                    raise ValueError(
                        "No documents are ingested. Run the ingestion script first."
                    )
                records = [
                    DocumentRecord(doc.doc_id, doc.title, doc.text)
                    for doc in documents
                ]
                started = time.perf_counter()
                _bm25_retriever = BM25(records)
                logger.info(
                    "Built BM25 index for %d documents in %.3f seconds",
                    len(records),
                    time.perf_counter() - started,
                )
    return _bm25_retriever


@app.on_event("startup")
def preload_resources() -> None:
    global _preload_complete, _preload_error
    if not _preload_enabled():
        return
    try:
        from scripts.embed.config import EmbeddingConfig

        if engine is None:
            raise RuntimeError("DATABASE_URL is not configured.")
        _get_vector_embedder(EmbeddingConfig.from_environment().model)
        with Session(engine) as session:
            _get_bm25_retriever(session)
        _preload_complete = True
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        SQLAlchemyError,
    ) as error:
        _preload_error = str(error)
        logger.exception("Application preload failed: %s", error)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Confirm that the API process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def readiness_check() -> dict[str, str]:
    if not _preload_enabled():
        raise HTTPException(
            status_code=503,
            detail=f"Preloading is disabled; set {PRELOAD_ENV}=true to enable readiness.",
        )
    if _preload_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Application preload failed: {_preload_error}",
        )
    if not _preload_complete:
        raise HTTPException(status_code=503, detail="Application preload is not complete.")
    return {"status": "ready"}


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
    mode: Annotated[Literal["bm25", "vector", "hybrid"], Query()] = "bm25",
) -> SearchResponse:
    """Search the ingested corpus with BM25, vector, or frozen hybrid retrieval."""
    if mode in {"vector", "hybrid"}:
        try:
            from scripts.embed.config import EmbeddingConfig

            config = EmbeddingConfig.from_environment()
            embedder = _get_vector_embedder(config.model)
            vector = VectorRetriever(session, embedder)
            if mode == "hybrid":
                import json

                hybrid_config = json.loads(
                    (ROOT_DIR / "config" / "hybrid_final.json").read_text(
                        encoding="utf-8"
                    )
                )
                retriever = HybridRetriever(
                    _get_bm25_retriever(session),
                    vector,
                    method=hybrid_config["method"],
                    candidate_depth=hybrid_config["depth"],
                    **hybrid_config["parameters"],
                )
            else:
                retriever = vector
            scores = dict(retriever.search(query, limit))
            documents = list(
                session.scalars(
                    select(Document).where(Document.doc_id.in_(scores))
                )
            )
            result_by_id = {document.doc_id: document for document in documents}
            return SearchResponse(
                query=query,
                mode=mode,
                results=[
                    SearchResult(
                        doc_id=doc_id,
                        title=result_by_id[doc_id].title,
                        text=result_by_id[doc_id].text,
                        score=score,
                    )
                    for doc_id, score in scores.items()
                    if doc_id in result_by_id
                ],
            )
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            logger.exception("Vector model or dependency failure: %s", error)
            raise HTTPException(
                status_code=503,
                detail=f"Vector model or dependency unavailable: {error}",
            ) from error
        except SQLAlchemyError as error:
            logger.exception("Vector retrieval database failure: %s", error)
            raise HTTPException(
                status_code=503,
                detail=f"Vector retrieval database unavailable: {error}",
            ) from error
    try:
        index = _get_bm25_retriever(session)
    except (SQLAlchemyError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The local database is unavailable."
                if isinstance(error, SQLAlchemyError)
                else str(error)
            ),
        ) from error
    scores = dict(index.search(query, limit))
    result_by_id = {record.doc_id: record for record in index.documents}
    results = [
        SearchResult(
            doc_id=doc_id,
            title=result_by_id[doc_id].title,
            text=result_by_id[doc_id].text,
            score=score,
        )
        for doc_id, score in scores.items()
    ]
    return SearchResponse(query=query, mode=mode, results=results)

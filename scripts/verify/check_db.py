import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]


def _scalar(connection, query: str):
    return connection.execute(text(query)).scalar_one()


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it in the root .env file."
        )

    engine = create_engine(database_url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as connection:
            orphan_chunks = _scalar(
                connection,
                """
                SELECT count(*)
                FROM chunks AS c
                LEFT JOIN documents AS d ON d.doc_id = c.doc_id
                WHERE d.doc_id IS NULL
                """,
            )
            orphan_qrels = _scalar(
                connection,
                """
                SELECT count(*)
                FROM qrels AS q
                LEFT JOIN documents AS d ON d.doc_id = q.doc_id
                LEFT JOIN queries AS bq ON bq.query_id = q.query_id
                WHERE d.doc_id IS NULL OR bq.query_id IS NULL
                """,
            )
            zero_chunk_documents = _scalar(
                connection,
                """
                SELECT count(*)
                FROM (
                    SELECT d.doc_id
                    FROM documents AS d
                    LEFT JOIN chunks AS c ON c.doc_id = d.doc_id
                    GROUP BY d.doc_id
                    HAVING count(c.id) = 0
                ) AS documents_without_chunks
                """,
            )
            duplicate_pairs = _scalar(
                connection,
                """
                SELECT count(*)
                FROM (
                    SELECT doc_id, chunk_index
                    FROM chunks
                    GROUP BY doc_id, chunk_index
                    HAVING count(*) > 1
                ) AS duplicates
                """,
            )
            empty_chunk_text = _scalar(
                connection,
                """
                SELECT count(*)
                FROM chunks
                WHERE btrim(text) = ''
                """,
            )
            distribution = connection.execute(
                text(
                    """
                    SELECT
                        min(chunk_count),
                        percentile_cont(0.5) WITHIN GROUP (
                            ORDER BY chunk_count
                        ),
                        percentile_cont(0.9) WITHIN GROUP (
                            ORDER BY chunk_count
                        ),
                        max(chunk_count),
                        avg((chunk_count >= 2)::int)
                    FROM (
                        SELECT d.doc_id, count(c.id)::int AS chunk_count
                        FROM documents AS d
                        LEFT JOIN chunks AS c ON c.doc_id = d.doc_id
                        GROUP BY d.doc_id
                    ) AS per_document
                    """
                )
            ).one()
            embedded_chunks = _scalar(
                connection,
                """
                SELECT count(*)
                FROM chunks
                WHERE embedding IS NOT NULL
                """,
            )
            unembedded_chunks = _scalar(
                connection,
                """
                SELECT count(*)
                FROM chunks
                WHERE embedding IS NULL
                """,
            )
            embedding_models = [
                row[0]
                for row in connection.execute(
                    text(
                        """
                        SELECT DISTINCT embedding_model
                        FROM chunks
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding_model
                        """
                    )
                )
            ]
            wrong_dimension = _scalar(
                connection,
                """
                SELECT count(*)
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND vector_dims(embedding) <> 768
                """,
            )
            non_finite = _scalar(
                connection,
                """
                SELECT count(*)
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND embedding::text ~ '(NaN|Infinity)'
                """,
            )
    finally:
        engine.dispose()

    print(f"Orphan chunks: {orphan_chunks}")
    print(f"Orphan qrels: {orphan_qrels}")
    print(f"Documents with zero chunks: {zero_chunk_documents}")
    print(f"Duplicate (document_id, chunk_index) pairs: {duplicate_pairs}")
    print(f"Empty or whitespace-only chunk text: {empty_chunk_text}")
    print(
        "Chunks per document: "
        f"min={distribution[0]}, median={distribution[1]}, "
        f"p90={distribution[2]}, max={distribution[3]}, "
        f"share with 2+={distribution[4]:.6f}"
    )
    print(f"Chunks with non-null embedding: {embedded_chunks}")
    print(f"Chunks with null embedding: {unembedded_chunks}")
    print(f"Embedding models: {embedding_models}")
    print(f"Embeddings with wrong dimension: {wrong_dimension}")
    print(f"Embeddings with non-finite values: {non_finite}")

    failures = {
        "orphan chunks": orphan_chunks,
        "orphan qrels": orphan_qrels,
        "documents with zero chunks": zero_chunk_documents,
        "duplicate pairs": duplicate_pairs,
        "empty chunk text": empty_chunk_text,
        "embeddings with wrong dimension": wrong_dimension,
        "embeddings with non-finite values": non_finite,
    }
    failed_checks = [name for name, count in failures.items() if count]
    if len(embedding_models) > 1:
        failed_checks.append("multiple embedding models")
    if failed_checks:
        print("Integrity checks failed: " + ", ".join(failed_checks))
        return 1

    print("Integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

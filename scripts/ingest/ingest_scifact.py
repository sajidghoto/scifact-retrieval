import csv
import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
from sqlalchemy import Connection, create_engine, delete
from sqlalchemy.dialects.postgresql import insert

from app.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text
from app.db.models import BenchmarkQuery, Chunk, Document, Qrel

DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
BATCH_SIZE = 500
T = TypeVar("T")


def find_dataset_files() -> tuple[Path, Path, Path]:
    """Return the local SciFact corpus, queries, and test qrels paths."""
    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels" / "test.tsv"

    required_paths = [corpus_path, queries_path, qrels_path]
    missing = [str(path) for path in required_paths if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Required SciFact files are missing:\n"
            + "\n".join(missing)
        )

    return corpus_path, queries_path, qrels_path


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )

            yield line_number, record


def _required_string(
    record: dict[str, object], key: str, path: Path, line_number: int
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Expected a non-empty string for {key!r} in "
            f"{path} at line {line_number}"
        )
    return value


def _iter_documents(path: Path) -> Iterator[dict[str, str]]:
    for line_number, record in _iter_jsonl(path):
        title = record.get("title", "")
        if not isinstance(title, str):
            raise ValueError(
                f"Expected a string for 'title' in {path} at line {line_number}"
            )

        yield {
            "doc_id": _required_string(record, "_id", path, line_number),
            "title": title,
            "text": _required_string(record, "text", path, line_number),
        }


def _iter_queries(path: Path) -> Iterator[dict[str, str]]:
    for line_number, record in _iter_jsonl(path):
        yield {
            "query_id": _required_string(record, "_id", path, line_number),
            "text": _required_string(record, "text", path, line_number),
        }


def _iter_qrels(path: Path) -> Iterator[dict[str, str | int]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_columns = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Expected TSV columns {sorted(required_columns)} in {path}"
            )

        for line_number, record in enumerate(reader, start=2):
            query_id = record["query-id"]
            doc_id = record["corpus-id"]
            raw_score = record["score"]
            if not query_id or not doc_id or raw_score is None:
                raise ValueError(
                    f"Incomplete qrel in {path} at line {line_number}"
                )

            try:
                score = int(raw_score)
            except ValueError as error:
                raise ValueError(
                    f"Invalid qrel score in {path} at line {line_number}: "
                    f"{raw_score!r}"
                ) from error

            yield {"query_id": query_id, "doc_id": doc_id, "score": score}


def _batches(records: Iterable[T]) -> Iterator[list[T]]:
    batch: list[T] = []
    for record in records:
        batch.append(record)
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []

    if batch:
        yield batch


def _upsert_documents(
    connection: Connection, records: list[dict[str, str]]
) -> None:
    statement = insert(Document).values(records)
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=[Document.doc_id],
            set_={
                "title": statement.excluded.title,
                "text": statement.excluded.text,
            },
        )
    )


def _upsert_queries(
    connection: Connection, records: list[dict[str, str]]
) -> None:
    statement = insert(BenchmarkQuery).values(records)
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=[BenchmarkQuery.query_id],
            set_={"text": statement.excluded.text},
        )
    )


def _upsert_qrels(
    connection: Connection, records: list[dict[str, str | int]]
) -> None:
    statement = insert(Qrel).values(records)
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=[Qrel.query_id, Qrel.doc_id],
            set_={"score": statement.excluded.score},
        )
    )


def _replace_chunks(
    connection: Connection, records: list[dict[str, str | int]]
) -> None:
    if not records:
        return

    doc_ids = {record["doc_id"] for record in records}
    connection.execute(delete(Chunk).where(Chunk.doc_id.in_(doc_ids)))
    connection.execute(insert(Chunk).values(records))


def _chunk_records(document: dict[str, str]) -> Iterator[dict[str, str | int]]:
    for chunk_index, text in enumerate(
        chunk_text(document["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    ):
        yield {
            "doc_id": document["doc_id"],
            "chunk_index": chunk_index,
            "text": text,
        }


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure it in the root .env file."
        )

    corpus_path, queries_path, qrels_path = find_dataset_files()
    engine = create_engine(database_url)

    try:
        with engine.begin() as connection:
            document_count = 0
            for batch in _batches(_iter_documents(corpus_path)):
                _upsert_documents(connection, batch)
                chunk_records = [
                    chunk
                    for document in batch
                    for chunk in _chunk_records(document)
                ]
                _replace_chunks(connection, chunk_records)
                document_count += len(batch)

            query_count = 0
            for batch in _batches(_iter_queries(queries_path)):
                _upsert_queries(connection, batch)
                query_count += len(batch)

            qrel_count = 0
            for batch in _batches(_iter_qrels(qrels_path)):
                _upsert_qrels(connection, batch)
                qrel_count += len(batch)
    finally:
        engine.dispose()

    print(f"Corpus documents processed: {document_count}")
    print(f"Queries processed: {query_count}")
    print(f"Test qrels processed: {qrel_count}")
    print(f"Chunk configuration: {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap")


if __name__ == "__main__":
    main()
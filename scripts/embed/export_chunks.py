"""Export stored chunks for out-of-process embedding."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))

from app.db.models import Chunk  # noqa: E402
from scripts.embed.config import EmbeddingConfig  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_checksum(rows: list[tuple[int, str]]) -> str:
    digest = hashlib.sha256()
    for chunk_id, text in rows:
        digest.update(f"{chunk_id}\t{text}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_export(path: Path, rows: list[tuple[int, str]]) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        jsonl_path = path.with_suffix(".jsonl")
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as file:
            for chunk_id, text in rows:
                file.write(json.dumps({"id": chunk_id, "text": text}, ensure_ascii=False))
                file.write("\n")
        return jsonl_path

    table = pa.table(
        {"id": [chunk_id for chunk_id, _ in rows], "text": [text for _, text in rows]}
    )
    pq.write_table(table, path)
    return path


def export_chunks(output: Path, sidecar: Path) -> dict[str, object]:
    load_dotenv(ROOT_DIR / ".env")
    config = EmbeddingConfig.from_environment()
    database_url = __import__("os").getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            rows = [
                (int(row.id), row.text)
                for row in session.scalars(select(Chunk).order_by(Chunk.id))
            ]
    finally:
        engine.dispose()

    output.parent.mkdir(parents=True, exist_ok=True)
    actual_output = _write_export(output, rows)
    metadata = {
        "row_count": len(rows),
        "file_sha256": _sha256(actual_output),
        "id_text_sha256": _rows_checksum(rows),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "dimension": config.dimension,
        "file": actual_output.name,
        "format": actual_output.suffix.lstrip("."),
        "checksum_definition": "SHA-256 of UTF-8 id\\ttext\\n lines in ascending id order",
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} chunks to {actual_output}")
    print(f"File SHA-256: {metadata['file_sha256']}")
    print(f"ID/text SHA-256: {metadata['id_text_sha256']}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data/exports/chunks.parquet")
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args(argv)
    sidecar = args.sidecar or args.output.with_name("chunks.meta.json")
    export_chunks(args.output, sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True
    )
    doc_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "doc_id", "chunk_index", name="uq_chunks_doc_chunk_index"
        ),
        Index("ix_chunks_doc_id", "doc_id"),
    )


class BenchmarkQuery(Base):
    __tablename__ = "queries"

    query_id: Mapped[str] = mapped_column(Text, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)


class Qrel(Base):
    __tablename__ = "qrels"

    query_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("queries.query_id", ondelete="CASCADE"),
        primary_key=True,
    )
    doc_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        primary_key=True,
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_qrels_doc_id", "doc_id"),
    )

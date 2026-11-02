"""Align embedding vectors and record their producing model."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR


revision: str = "8a1f2c3d4e5f"
down_revision: Union[str, Sequence[str], None] = "0638444e2212"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024) "
        "USING embedding::vector"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector "
        "USING embedding::vector"
    )
    op.drop_column("chunks", "embedding_model")

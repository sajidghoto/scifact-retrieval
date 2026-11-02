"""Switch stored embeddings to bge-base dimensions."""

from typing import Sequence, Union

from alembic import op


revision: str = "9b2c3d4e5f6a"
down_revision: Union[str, Sequence[str], None] = "8a1f2c3d4e5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE chunks SET embedding = NULL, embedding_model = NULL"
    )
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(768) "
        "USING embedding::vector"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE chunks SET embedding = NULL, embedding_model = NULL"
    )
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024) "
        "USING embedding::vector"
    )

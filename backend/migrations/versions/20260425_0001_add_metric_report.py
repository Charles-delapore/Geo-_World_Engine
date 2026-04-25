"""add metric_report column

Revision ID: 20260425_0001
Revises: 20260414_0001
Create Date: 2026-04-25
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260425_0001"
down_revision = "20260414_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("metric_report", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("tasks", "metric_report")

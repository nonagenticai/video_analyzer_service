"""initial analysis_jobs table

Revision ID: 001_initial
Revises:
Create Date: 2026-04-13

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(64), unique=True, nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(64), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float, nullable=False, server_default="0"),
        sa.Column("video_path", sa.Text, nullable=True),
        sa.Column("video_url", sa.Text, nullable=True),
        sa.Column("upload_filename", sa.Text, nullable=True),
        sa.Column("request_config", JSONB, nullable=True),
        sa.Column("verification_criteria", JSONB, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("fps", sa.Float, nullable=True),
        sa.Column("frame_count", sa.Integer, nullable=True),
        sa.Column("has_audio", sa.Integer, nullable=True),
        sa.Column("transcript_data", JSONB, nullable=True),
        sa.Column("frame_analyses", JSONB, nullable=True),
        sa.Column("scene_boundaries", JSONB, nullable=True),
        sa.Column("quality_metrics", JSONB, nullable=True),
        sa.Column("verification_result", JSONB, nullable=True),
        sa.Column("artifacts", JSONB, nullable=True),
        sa.Column("vision_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_analysis_jobs_tenant_id", "analysis_jobs", ["tenant_id"], unique=False)
    op.create_index(
        "ix_analysis_jobs_tenant_status",
        "analysis_jobs",
        ["tenant_id", "status"],
        unique=False,
    )
    op.create_index("ix_analysis_jobs_created_at", "analysis_jobs", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_analysis_jobs_created_at", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_tenant_status", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_tenant_id", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")

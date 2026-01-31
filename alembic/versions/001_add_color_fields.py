"""Add skin_color and dominant_color fields

Revision ID: 001_add_color_fields
Revises:
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_add_color_fields'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add skin_color column to avatars table
    op.add_column('avatars', sa.Column('skin_color', postgresql.JSON(astext_type=sa.Text()), nullable=True))

    # Add dominant_color column to garments table
    op.add_column('garments', sa.Column('dominant_color', postgresql.JSON(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('garments', 'dominant_color')
    op.drop_column('avatars', 'skin_color')

"""Seed sample sales rows and a default admin into the database."""

from __future__ import annotations

from sqlalchemy.orm import Session

from .models import SalesRecord, User
from .security import hash_password


_SAMPLE_SALES: list[tuple[str, str, float]] = [
    ("books", "2024-01", 120.0),
    ("books", "2024-02", 180.5),
    ("books", "2024-03", 210.0),
    ("books", "2024-04", 175.25),
    ("electronics", "2024-01", 540.0),
    ("electronics", "2024-02", 612.75),
    ("electronics", "2024-03", 488.0),
    ("electronics", "2024-04", 705.5),
    ("clothing", "2024-01", 230.0),
    ("clothing", "2024-02", 245.0),
    ("clothing", "2024-03", 198.0),
    ("clothing", "2024-04", 312.0),
]


def seed_sample_data(db: Session) -> None:
    """Insert sample sales rows and a default admin user if absent."""
    if db.query(SalesRecord).first() is None:
        for category, month, amount in _SAMPLE_SALES:
            db.add(SalesRecord(category=category, month=month, amount=amount))

    if db.query(User).filter(User.username == "admin").first() is None:
        db.add(
            User(
                username="admin",
                email="admin@example.com",
                hashed_password=hash_password("adminpass"),
                role="admin",
            )
        )

    db.commit()

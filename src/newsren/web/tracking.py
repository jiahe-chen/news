"""Privacy-preserving engagement tracking (only signed-in readers, only story/brief views)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from newsren.models import PageView, User


def record_view(
    db: Session, user: User | None, path: str, *, story_id: int | None = None, brief_id: int | None = None
) -> None:
    """Record a story/brief view for the 'return visits to followed stories' metric."""
    if user is None:
        return
    db.add(PageView(user_id=user.id, path=path[:300], story_id=story_id, brief_id=brief_id))

"""Database connection — thin re-export of w2v_common."""

from w2v_common.database.connection import close_engine, get_engine, health_check

__all__ = ["close_engine", "get_engine", "health_check"]

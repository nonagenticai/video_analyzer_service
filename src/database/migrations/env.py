"""Alembic migration environment for video_analyzer_service."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/video_analyzer",
)


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# Session-level advisory lock key for the whole migration run. This service is
# deployed at ``replicas: 2`` with RollingUpdate, and each pod runs its own migrate
# initContainer — so two alembic processes can start against the same database at the
# same time. Without serialisation the loser of a DDL race dies (``DuplicateColumn`` /
# ``DuplicateObject``) and the pod CrashLoopBackOffs. The compiler hit exactly this in
# its migration 004, whose docstring still assumes "the initContainer applies this
# after the old pod is torn down", which is false at >1 replica.
#
# ``pg_advisory_lock`` (session-scoped, not ``_xact_``) is used deliberately: alembic
# opens and commits its own transactions per revision, so a transaction-scoped lock
# would be released after the first one. ``NullPool`` + ``dispose()`` in
# ``run_async_migrations`` closes the session, which releases the lock even if the
# process dies mid-run. The loser BLOCKS rather than failing — that is the point.
_MIGRATION_LOCK_KEY = "video_analyzer_migrations"


def do_run_migrations(connection) -> None:
    # The ``commit()`` calls around the lock statements are LOAD-BEARING, not tidiness.
    # Under SQLAlchemy 2.0 a bare ``exec_driver_sql`` on a ``connect()``-ed connection
    # opens an implicit transaction. If one is left open here, alembic's own
    # ``begin_transaction()`` NESTS inside it instead of owning the outer transaction —
    # so ``run_migrations`` never commits, the connection closes, everything ROLLS BACK,
    # and ``alembic upgrade head`` still exits 0 having applied nothing. That failure was
    # observed while writing this: every revision logged "Running upgrade ..." and the
    # target database ended up with no ``alembic_version`` table at all. A silent no-op
    # migration on a service whose initContainer gates the rollout is the worst possible
    # shape of this bug, so it is committed explicitly and verified below.
    #
    # A session-level advisory lock is unaffected by COMMIT (only ``pg_advisory_xact_lock``
    # is transaction-scoped), so committing here does not release it.
    connection.exec_driver_sql(f"SELECT pg_advisory_lock(hashtext('{_MIGRATION_LOCK_KEY}'))")
    connection.commit()
    try:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    finally:
        connection.exec_driver_sql(f"SELECT pg_advisory_unlock(hashtext('{_MIGRATION_LOCK_KEY}'))")
        connection.commit()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = DATABASE_URL
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 10},
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings

# Import Base and ALL models so Alembic can see them for autogenerate.
# If you create a new model and forget to import it here,
# autogenerate will not detect it and won't generate the migration.
from app.db.models import Base  # noqa: F401 - import triggers model registration

config = context.config
fileConfig(config.config_file_name)

# This is what Alembic compares against the live database to generate diffs.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection (generates SQL script only).
    Used when you want to review SQL before applying it.
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """
    Run migrations against a live database.
    We use create_async_engine here because our driver (asyncpg) is async.
    run_sync() bridges the async connection to Alembic's sync migration runner.
    """
    connectable = create_async_engine(settings.database_url)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

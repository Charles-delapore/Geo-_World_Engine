from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from app.config import settings


def main() -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

    engine = create_engine(settings.DATABASE_URL, future=True)
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    with engine.begin() as connection:
        has_alembic_version = "alembic_version" in table_names
        has_legacy_tables = {"tasks", "task_state_transitions"} & table_names

        if has_legacy_tables and not has_alembic_version:
            command.stamp(config, "head")
            return

        if has_alembic_version:
            connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))

    command.upgrade(config, "head")


if __name__ == "__main__":
    main()

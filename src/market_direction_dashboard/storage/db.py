from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def resolve_sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Expected a sqlite:/// URL when resolving a local sqlite path.")
    raw_path = database_url.removeprefix("sqlite:///")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def open_database(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///"):
        db_path = resolve_sqlite_path(database_url)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    engine = create_engine(database_url, future=True)
    
    if database_url.startswith("sqlite:///"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
            
    return engine


def create_schema(target: Engine | Session) -> None:
    engine = target if isinstance(target, Engine) else target.get_bind()
    Base.metadata.create_all(engine)
    _migrate_daily_prediction_runs(engine)
    _migrate_graph_build_tasks(engine)
    _migrate_daily_graph_summary(engine)
    _migrate_daily_graph_delta(engine)


@contextmanager
def database_session(database_url: str):
    engine = open_database(database_url)
    SessionFactory = sessionmaker(bind=engine, future=True)
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _migrate_daily_prediction_runs(engine: Engine) -> None:
    inspector = inspect(engine)
    if "daily_prediction_runs" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("daily_prediction_runs")}
    additions = {
        "feature_snapshot_version": "VARCHAR(128)",
        "model_stack_version": "VARCHAR(128)",
        "calibration_version": "VARCHAR(128)",
        "regime_slice": "VARCHAR(64)",
        "agreement_score": "FLOAT",
        "pipeline_stage_status": "TEXT",
        "stage_diagnostics": "TEXT",
    }
    with engine.begin() as connection:
        for column_name, ddl in additions.items():
            if column_name in existing:
                continue
            connection.execute(text(f"ALTER TABLE daily_prediction_runs ADD COLUMN {column_name} {ddl}"))


def _migrate_graph_build_tasks(engine: Engine) -> None:
    inspector = inspect(engine)
    if "graph_build_tasks" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("graph_build_tasks")}
    additions = {
        "progress_detail": "TEXT",
        "stage_started_at": "VARCHAR(64)",
        "last_progress_at": "VARCHAR(64)",
        "telemetry_json": "TEXT",
    }
    with engine.begin() as connection:
        for column_name, ddl in additions.items():
            if column_name in existing:
                continue
            connection.execute(text(f"ALTER TABLE graph_build_tasks ADD COLUMN {column_name} {ddl}"))


def _migrate_daily_graph_summary(engine: Engine) -> None:
    inspector = inspect(engine)
    if "daily_graph_summary" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("daily_graph_summary")}
    additions = {
        "feature_group": "VARCHAR(64)",
        "schema_version": "VARCHAR(64)",
        "generation_run_id": "VARCHAR(128)",
    }
    with engine.begin() as connection:
        for column_name, ddl in additions.items():
            if column_name in existing:
                continue
            connection.execute(text(f"ALTER TABLE daily_graph_summary ADD COLUMN {column_name} {ddl}"))


def _migrate_daily_graph_delta(engine: Engine) -> None:
    inspector = inspect(engine)
    if "daily_graph_delta" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("daily_graph_delta")}
    additions = {
        "feature_group": "VARCHAR(64)",
        "schema_version": "VARCHAR(64)",
        "generation_run_id": "VARCHAR(128)",
    }
    with engine.begin() as connection:
        for column_name, ddl in additions.items():
            if column_name in existing:
                continue
            connection.execute(text(f"ALTER TABLE daily_graph_delta ADD COLUMN {column_name} {ddl}"))

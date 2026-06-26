"""SQLite-backed HistoryStore via aiosqlite."""

from __future__ import annotations

import json
import os
from datetime import datetime

import aiosqlite

from src.evaluation.interfaces.evaluator import EvalResult
from src.interfaces.history import HistoryRecord, HistoryStore

_DB_PATH = os.getenv("RAG_HISTORY_DB_PATH", "history.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS query_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    sources     TEXT NOT NULL,
    eval_result TEXT,
    retries     INTEGER NOT NULL DEFAULT 0
)
"""


class SQLiteHistory(HistoryStore):
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DB_PATH

    async def _init(self, conn: aiosqlite.Connection) -> None:
        conn.row_factory = aiosqlite.Row
        await conn.execute(_CREATE_TABLE)
        await conn.commit()

    async def save(self, record: HistoryRecord) -> None:
        eval_json = record.eval_result.model_dump_json() if record.eval_result else None
        async with aiosqlite.connect(self._db_path) as conn:
            await self._init(conn)
            await conn.execute(
                """
                INSERT INTO query_history (timestamp, question, answer, sources, eval_result, retries)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.question,
                    record.answer,
                    json.dumps(record.sources),
                    eval_json,
                    record.retries,
                ),
            )
            await conn.commit()

    async def list(self, limit: int = 50) -> list[HistoryRecord]:
        async with aiosqlite.connect(self._db_path) as conn:
            await self._init(conn)
            async with conn.execute(
                "SELECT * FROM query_history ORDER BY id DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()

        records = []
        for row in rows:
            eval_result = (
                EvalResult.model_validate_json(row["eval_result"]) if row["eval_result"] else None
            )
            records.append(
                HistoryRecord(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    question=row["question"],
                    answer=row["answer"],
                    sources=json.loads(row["sources"]),
                    eval_result=eval_result,
                    retries=row["retries"],
                )
            )
        return records


_store: SQLiteHistory | None = None


def get_history_store() -> SQLiteHistory:
    global _store
    if _store is None:
        _store = SQLiteHistory()
    return _store

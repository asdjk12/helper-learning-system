from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable


class SQLiteKnowledgeStore:
    def __init__(self, db_path: str = "data/knowledge.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    domain TEXT,
                    status TEXT NOT NULL DEFAULT 'stored',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256
                ON documents(sha256);

                CREATE TABLE IF NOT EXISTS document_pages (
                    page_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                    ON DELETE CASCADE,

                    UNIQUE(document_id, page_number)
                );

                CREATE INDEX IF NOT EXISTS idx_document_pages_document_id
                ON document_pages(document_id);

                CREATE INDEX IF NOT EXISTS idx_document_pages_text_hash
                ON document_pages(text_hash);
                """
            )

    def upsert_document(
        self,
        document_metadata: dict,
        page_count: int = 0,
        domain: str | None = None,
        status: str = "stored",
    ) -> None:
        required_fields = ["document_id", "filename", "file_path", "sha256", "file_size"]
        missing_fields = [
            field for field in required_fields if not document_metadata.get(field)
        ]
        if missing_fields:
            raise ValueError(f"Missing document metadata fields: {missing_fields}")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    filename,
                    file_path,
                    sha256,
                    file_size,
                    page_count,
                    domain,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename = excluded.filename,
                    file_path = excluded.file_path,
                    sha256 = excluded.sha256,
                    file_size = excluded.file_size,
                    page_count = excluded.page_count,
                    domain = excluded.domain,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    document_metadata["document_id"],
                    document_metadata["filename"],
                    document_metadata["file_path"],
                    document_metadata["sha256"],
                    int(document_metadata["file_size"]),
                    int(page_count),
                    domain,
                    status,
                ),
            )

    def store_document_pages(self, document_id: str, pages: Iterable[dict]) -> int:
        rows = []
        for page in pages:
            page_number = int(page["page_number"])
            text = (page.get("text") or "").strip()
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rows.append(
                (
                    f"{document_id}:page:{page_number}",
                    document_id,
                    page_number,
                    text,
                    text_hash,
                    len(text),
                )
            )

        if not rows:
            return 0

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_pages (
                    page_id,
                    document_id,
                    page_number,
                    text,
                    text_hash,
                    char_count
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, page_number) DO UPDATE SET
                    text = excluded.text,
                    text_hash = excluded.text_hash,
                    char_count = excluded.char_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )

        return len(rows)

    def get_document_pages(self, document_id: str) -> list[dict]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT page_id, document_id, page_number, text, text_hash, char_count
                FROM document_pages
                WHERE document_id = ?
                ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_document_by_sha256(self, sha256: str) -> dict | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE sha256 = ?
                """,
                (sha256,),
            ).fetchone()

        return dict(row) if row else None

    def get_document(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return dict(row) if row else None

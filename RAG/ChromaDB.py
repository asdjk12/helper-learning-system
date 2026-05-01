from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import chromadb


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "course_materials",
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else self.default_persist_dir()
        self.persist_dir = self.persist_dir.resolve()
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Course materials for RAG"},
        )

    @staticmethod
    def default_persist_dir() -> Path:
        return Path(__file__).resolve().parent / "data" / "chroma"

    def add(self, document_id: str, blocks: list[dict]) -> None:
        ids = []
        documents = []
        metadatas = []

        for index, content in enumerate(blocks):
            content_id = content.get("block_id", f"{document_id}:block:{index}")

            ids.append(content_id)
            documents.append(self._build_document_text(content))
            metadatas.append(
                {
                    "document_id": document_id,
                    "block_id": content_id,
                    "filename": content.get("filename", ""),
                    "title": content.get("title", ""),
                    "source_pages": content.get("source_pages", ""),
                    "block_type": content.get("block_type", "summary"),
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def _build_document_text(self, block: dict) -> str:
        formulas = block.get("formulas", [])
        methods = block.get("methods", [])
        examples = block.get("examples", [])

        formulas_text = "\n".join(
            f"- {item.get('name', '')}: {item.get('expression', '')} {item.get('explanation', '')}"
            if isinstance(item, dict) else f"- {item}"
            for item in formulas
        )

        methods_text = "\n".join(
            f"- {item.get('name', '')}: {'; '.join(item.get('steps', []))}"
            if isinstance(item, dict) else f"- {item}"
            for item in methods
        )

        examples_text = "\n".join(
            f"- Q: {item.get('question', '')}\n  A: {item.get('answer', '')}"
            if isinstance(item, dict) else f"- {item}"
            for item in examples
        )

        return "\n\n".join(
            [
                f"Title: {block.get('title', '')}",
                f"Type: {block.get('block_type', '')}",
                f"Summary:\n{block.get('summary', '')}",
                "Concepts:\n" + ", ".join(block.get("concepts", [])),
                "Key points:\n" + "\n".join(f"- {point}" for point in block.get("key_points", [])),
                "Evidence:\n" + "\n".join(f"- {item}" for item in block.get("evidence", [])),
                "Formulas:\n" + formulas_text,
                "Methods:\n" + methods_text,
                "Examples:\n" + examples_text,
            ]
        )

    def get_block(self, block_id):
        return self.collection.get(ids=[block_id])

    def search(self, query, n_results=5):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

    def delete_block(self, block_id):
        self.collection.delete(ids=block_id)

    def get_all_blocks(self):
        return self.collection.get()

    def remove_all(self):
        """
        Remove all Chroma content for this store and delete the persisted files.

        After this call the current ChromaVectorStore instance should be discarded.
        Create a new ChromaVectorStore() before adding/searching again.
        """
        persist_dir = self.persist_dir

        if self.collection is not None:
            all_blocks = self.collection.get()
            ids = all_blocks.get("ids", [])

            if ids:
                self.collection.delete(ids=ids)

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            raise ValueError("delete_collection error")

        self.collection = None
        self.client = None
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            raise ValueError("SharedSystemClient error")

        gc.collect()

        persist_dir = persist_dir.resolve()
        if not persist_dir.exists():
            return

        if persist_dir.name.lower() != "chroma":
            raise ValueError(
                "Refusing to delete Chroma persist directory because the final "
                f"path segment is not 'chroma': {persist_dir}"
            )

        last_error = None
        for _ in range(5):
            try:
                shutil.rmtree(persist_dir)
                return
            except PermissionError as error:
                last_error = error
                time.sleep(0.2)

        raise RuntimeError(
            "Failed to delete Chroma persist directory. "
            "A Python process may still be holding the database files open: "
            f"{persist_dir}"
        ) from last_error

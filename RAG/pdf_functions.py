import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlparse

import requests

try:
    from ..prompt import _repair_llm_json, yield_summarise
    from ..llm import model_init
    from .ChromaDB import ChromaVectorStore
    from pypdf import PdfReader
except ImportError:
    from prompt import _repair_llm_json, yield_summarise
    from llm import model_init
    from ChromaDB import ChromaVectorStore
    from PyPDF2 import PdfReader


def _parse_llm_json(content):
    content = (content or "").strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError as error:
            preview = content[:500].replace("\n", "\\n")
            raise ValueError(
                "LLM response is not valid JSON. "
                f"Preview: {preview}"
            ) from error


def _validate_blocks_schema(data, group):
    source_pages = group.get("source_pages", "<unknown_pages>")

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid LLM schema for pages {source_pages}: "
            f"top-level JSON must be an object, got {type(data).__name__}"
        )

    if "blocks" not in data:
        keys = list(data.keys())
        raise ValueError(
            f"Invalid LLM schema for pages {source_pages}: "
            f"missing 'blocks' key. Got keys: {keys}"
        )

    blocks = data["blocks"]

    if not isinstance(blocks, list):
        raise ValueError(
            f"Invalid LLM schema for pages {source_pages}: "
            f"'blocks' must be a list, got {type(blocks).__name__}"
        )

    if not blocks:
        raise ValueError(
            f"Invalid LLM schema for pages {source_pages}: "
            "'blocks' is empty"
        )

    required_string_fields = ["block_id", "title", "block_type", "summary"]
    
    # LLM 总结字段
    required_list_fields = [
        "concepts",
        "key_points",
        "evidence",
    ]
    optional_list_fields = [
        "formulas",
        "methods",
        "examples",
    ]

    # pdf 原文返回字段
    allowed_block_types = {
        "concept",
        "method",
        "formula",
        "example",
        "comparison",
        "history",
        "summary",
    }

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(
                f"Invalid LLM schema for pages {source_pages}: "
                f"block {index} must be an object, got {type(block).__name__}"
            )

        for field in required_string_fields:
            if not isinstance(block.get(field), str) or not block.get(field).strip():
                raise ValueError(
                    f"Invalid LLM schema for pages {source_pages}: "
                    f"block {index} missing non-empty string field '{field}'"
                )

        block_type = block.get("block_type", "").strip()
        if block_type not in allowed_block_types:
            block["block_type"] = "summary"

        for field in required_list_fields:
            if not isinstance(block.get(field), list) or not block[field]:
                raise ValueError(
                    f"Invalid LLM schema for pages {source_pages}: "
                    f"block {index} field '{field}' must be a non-empty list"
                )

        for field in optional_list_fields:
            if field not in block:
                block[field] = []
            elif not isinstance(block[field], list):
                raise ValueError(
                    f"Invalid LLM schema for pages {source_pages}: "
                    f"block {index} field '{field}' must be a list, "
                    f"got {type(block[field]).__name__}"
                )

    return blocks


def download_pdf_to_temp(file_url, temp_dir):
    # 临时文件下载
    if not file_url:
        raise ValueError("file_url is required")

    file_url = str(file_url)
    parsed_url = urlparse(file_url)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if parsed_url.scheme in ("http", "https"):
        filename = Path(unquote(parsed_url.path)).name or "uploaded.pdf"
        temp_pdf_path = temp_dir / filename

        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()

        with temp_pdf_path.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)

        return str(temp_pdf_path)

    source_path = Path(file_url).expanduser()
    if source_path.exists():
        temp_pdf_path = temp_dir / source_path.name
        shutil.copy2(source_path, temp_pdf_path)
        return str(temp_pdf_path)

    raise ValueError(f"Unsupported PDF source: {file_url}")


def calculate_file_sha256(file_path):
    # sha256计算，查重需要
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_pdf_file(temp_pdf_path, document_id, storage_dir="storage/pdfs"):
    """
    Keep the raw PDF on disk and return lightweight metadata for databases.
    The PDF bytes should not be inserted into Chroma or future SQL tables.
    """
    pdf_path = Path(temp_pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    sha256 = calculate_file_sha256(pdf_path)
    target_path = storage_path / f"{sha256}.pdf"

    if not target_path.exists():
        shutil.copy2(pdf_path, target_path)

    return {
        "document_id": document_id,
        "filename": pdf_path.name,
        "file_path": str(target_path),
        "sha256": sha256,
        "file_size": pdf_path.stat().st_size,
    }


def read_pdf_pages(temp_pdf_path):
    """
    Read PDF text page by page.

    Yields:
        {
            "filename": "lecture.pdf",
            "page_number": 1,
            "text": "..."
        }
    """
    pdf_path = Path(temp_pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with pdf_path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)

        for index, page in enumerate(reader.pages):
            yield {
                "filename": pdf_path.name,
                "page_number": index + 1,
                "text": (page.extract_text() or "").strip(),
            }


def group_pages(pages, max_tokens=1800):
    """
    Group neighboring PDF pages before semantic block extraction.

    This is not the final embedding chunk. It is only a bounded input window
    for the LLM, which later extracts one or more semantic knowledge blocks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    page_group = []
    current_tokens = 0

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        page_tokens = max(1, len(text) // 4)

        if page_group and current_tokens + page_tokens > max_tokens:
            yield {
                "filename": page_group[0]["filename"],
                "source_pages": f"{page_group[0]['page_number']}-{page_group[-1]['page_number']}",
                "text": "\n\n".join(
                    f"[Page {item['page_number']}]\n{item['text']}"
                    for item in page_group
                ),
            }

            page_group = []
            current_tokens = 0

        page_group.append(page)
        current_tokens += page_tokens

    if page_group:
        yield {
            "filename": page_group[0]["filename"],
            "source_pages": f"{page_group[0]['page_number']}-{page_group[-1]['page_number']}",
            "text": "\n\n".join(
                f"[Page {item['page_number']}]\n{item['text']}"
                for item in page_group
            ),
        }


def summarize_group_with_llm(group, model=None, max_attempts=3):
    """
    Summarize one page group into the knowledge-block JSON format.
    Retry generation up to max_attempts times, with one repair attempt
    inside each generation attempt.
    """
    client, default_model = model_init()
    model = model or default_model
    prompt = yield_summarise(group=group)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        print(f"LLM attempt {attempt}/{max_attempts} for pages {group['source_pages']}")
        raw_content = ""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )

            raw_content = response.choices[0].message.content
            data = _parse_llm_json(raw_content)
            blocks = _validate_blocks_schema(data, group)

        except ValueError as error:
            last_error = error
            print(
                f"Repair attempt {attempt}/{max_attempts} for pages "
                f"{group['source_pages']}"
            )

            try:
                repaired_content = _repair_llm_json(
                    client=client,
                    model=model,
                    broken_content=raw_content,
                    error=error,
                )
                repaired_data = _parse_llm_json(repaired_content)
                blocks = _validate_blocks_schema(repaired_data, group)
            except Exception as repair_error:
                last_error = repair_error
                if attempt == max_attempts:
                    raise
                continue

        except Exception as error:
            last_error = error
            if attempt == max_attempts:
                raise
            continue

        for index, block in enumerate(blocks):
            block.setdefault(
                "block_id",
                f"{Path(group['filename']).stem}:pages:{group['source_pages']}:block:{index}",
            )
            block["source_pages"] = group["source_pages"]
            block["filename"] = group["filename"]

        return blocks

    raise ValueError(
        f"Failed to summarize pages {group['source_pages']} "
        f"after {max_attempts} attempts"
    ) from last_error


def store_blocks(document_id, blocks, store=None):
    if not blocks:
        raise ValueError(f"No blocks to store for document_id={document_id}")

    store = store or ChromaVectorStore()
    store.add(document_id=document_id, blocks=blocks)
    return len(blocks)

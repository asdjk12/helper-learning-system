import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
import requests

try:
    from ..prompt import _repair_llm_json, yield_summarise
except ImportError:
    from prompt import _repair_llm_json, yield_summarise

try:
    from ..llm import model_init
except ImportError:
    from llm import model_init

try:
    from .ChromaDB import ChromaVectorStore
except ImportError:
    from ChromaDB import ChromaVectorStore

try:
    from pypdf import PdfReader
except ImportError:
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

    required_string_fields = ["title", "summary"]
    list_fields = ["key_points", "formulas", "rules", "methods", "examples"]

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

        for field in list_fields:
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
    """
    下载pdf至临时文件?    """
    if not file_url:
        raise ValueError("未收到file_url")

    file_url = str(file_url)
    parsed_url = urlparse(file_url)         # 拆解url
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if parsed_url.scheme in ("http", "https"):
        filename = Path(unquote(parsed_url.path)).name or "uploaded.pdf"
        temp_pdf_path = temp_dir / filename

        # 读取url并查看状态?        
        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()

        with temp_pdf_path.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)

        return str(temp_pdf_path)


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
    在将页面发送给LLM之前，请先将它们分组。
    这样可以保持源页码清晰，避免出现一大堆提示信息。    """
    if max_tokens <= 0:
        raise ValueError("group_size须为正整数")

    page_group = []
    current_tokens = 0

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        page_tokens = max(1, len(text) // 4)

        # token 验证限制防止过大
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

    client,model = model_init()

    prompt = yield_summarise(group=group)
    last_error = None

    # re-generation 机制
    for attempt in range(1, max_attempts + 1):
        print(f"LLM attempt {attempt}/{max_attempts} for pages {group['source_pages']}")    # 追踪
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
    # 确认只在block有效时入库
    if not blocks:
        raise ValueError(f"No blocks to store for document_id={document_id}")

    store = store or ChromaVectorStore()    # 初始化DB 
    store.add(document_id=document_id, blocks=blocks)
    return len(blocks)

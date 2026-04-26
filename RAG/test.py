# 这个文件主要存放RAG 过程中每个function的功能性测试

import tempfile
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pprint import pformat
from datetime import datetime
try:
    from .ChromaDB import ChromaVectorStore
    from .pdf_functions import *
except ImportError:
    from ChromaDB import ChromaVectorStore
    from pdf_functions import *

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

def DB_store_testing(file_url, document_id=None, max_tokens=1800, model=None):
    """
    Test full PDF -> LLM summary -> ChromaDB storage pipeline.
    """

    if document_id is None:
        parsed_url = urlparse(str(file_url))
        filename = Path(unquote(parsed_url.path)).stem if parsed_url.scheme else Path(file_url).stem
        document_id = filename or "test_document"

    store = ChromaVectorStore()
    total_groups = 0
    total_blocks = 0

    with tempfile.TemporaryDirectory(prefix="rag_pdf_") as temp_dir:
        temp_pdf_path = download_pdf_to_temp(
            file_url=file_url,
            temp_dir=temp_dir,
        )

        pages = list(read_pdf_pages(temp_pdf_path))
        groups = group_pages(pages, max_tokens=max_tokens)

        for group in groups:
            total_groups += 1

            blocks = summarize_group_with_llm(
                group=group,
                model=model,
            )

            stored_count = store_blocks(
                document_id=document_id,
                blocks=blocks,
                store=store,
            )

            total_blocks += stored_count

            print(
                f"Stored group {total_groups}: "
                f"pages {group['source_pages']}, "
                f"{stored_count} blocks"
            )

    return {
        "document_id": document_id,
        "groups": total_groups,
        "blocks": total_blocks,
    }

# 


def show_all_blocks(temp_dir=r"D:\Leetcode\MultipleAgentSystem\HelperSystem\RAG\temp_dir"):
    # 输出DB内容至txt文件 并 保存至temp_dir本地
    store = ChromaVectorStore()
    result = store.get_all_blocks()

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_file = temp_dir / f"all_blocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    ids = result.get("ids", [])
    metadatas = result.get("metadatas", [])
    documents = result.get("documents", [])

    with output_file.open("w", encoding="utf-8") as f:
        f.write("=== All Blocks In ChromaDB ===\n\n")

        for index, block_id in enumerate(ids):
            f.write(f"Block {index + 1}\n")
            f.write(f"ID: {block_id}\n")
            f.write(f"Metadata: {metadatas[index] if index < len(metadatas) else {}}\n")
            f.write(f"Document:\n{documents[index] if index < len(documents) else ''}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Database content saved to: {output_file}")
    return str(output_file)

def remove_all():
    store = ChromaVectorStore()
    store.remove_all()

if __name__ == "__main__":
    # remove_all()
    
    # 读取/汇总/保存
    result = DB_store_testing(
        file_url="https://raw.githubusercontent.com/asdjk12/pdf/main/Week%207%20-%20Neural%20Machine%20Translation.pdf")

    # print(result)
    # show_all_blocks()

    
    

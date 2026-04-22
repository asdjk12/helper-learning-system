from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

class ChromaVectorStore:
    def __init__(
            self,
            persist_dir: str = "data/chroma",   # ChromaDB 数据保存在本地哪个文件夹里
            collection_name: str = "course_materials",  # 数据库里的“表名”。
            ) -> None:
        
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir)) # 初始化客户端
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Course materials for RAG"},
        )
    
    def add(self, document_id:str,
                  blocks: list[dict],):
        """
        Example of sub-block:
            {
            "block_id": "week1_supervised_learning",
            "title": "Supervised Learning",
            "summary": "This block explains supervised learning, where a model learns from labelled examples to predict outputs for new inputs.",
            "key_points": [
                "Supervised learning uses input-output pairs.",
                "The goal is to learn a mapping from inputs to outputs.",
                "Common tasks include classification and regression."
            ],
            "formulas": [
                {
                "name": "Mean Squared Error",
                "expression": "MSE = (1/n) * sum((y_i - y_hat_i)^2)",
                "explanation": "Measures the average squared difference between actual and predicted values."
                }
            ],
            "rules": [
                "Use classification when the output is a category.",
                "Use regression when the output is a continuous value."
            ],
            "methods": [
                {
                "name": "Train-test split",
                "steps": [
                    "Split the dataset into training and testing sets.",
                    "Train the model on the training set.",
                    "Evaluate performance on the testing set."
                ]
                }
            ],
            "examples": [
                {
                "question": "Predict whether an email is spam or not.",
                "answer": "This is a classification problem because the output is a category."
                }
            ],
            "source_pages": "3-6",
            "document_id": "week1_slides",
            "filename": "week1.pdf"
            }

            summary    -> 帮助快速理解这一块讲什么
            key_points -> 帮助抓重点
            formulas   -> 回答计算题、数学推导题
            rules      -> 回答判断题、选择题、适用场景题
            methods    -> 回答“怎么做”的步骤题
            examples   -> 帮助 Tutor Agent 举例解释
        """

        # 添加内容至数据库
        ids = []
        documents = []
        metadatas = []

        for index,content in enumerate(blocks):
            # 获取一个block内的更小sub-block ID
            content_id = content.get("block_id", f"{document_id}:block:{index}")

            ids.append(content_id)  

            document_text = self._build_document_text(content)      # 片段evidence
            documents.append(document_text)     # 追踪原文

            # 
            metadatas.append({
                "document_id": document_id,
                "block_id": content_id,
                "filename": content.get("filename", ""),
                "title": content.get("title", ""),
                "source_pages": content.get("source_pages", ""),
                "block_type": content.get("block_type", "summary"),
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
    
    def _build_document_text(self, block):
        # 获取block信息 为 存进DB 做提取
        key_points = block.get("key_points", [])
        concepts = block.get("concepts", [])
        formulas = block.get("formulas", [])
        rules = block.get("rules", [])
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

        return "\n\n".join([
            f"Title: {block.get('title', '')}",
            f"Summary:\n{block.get('summary', '')}",
            "Key points:\n" + "\n".join(f"- {point}" for point in key_points),
            "Concepts:\n" + ", ".join(concepts),
            "Formulas:\n" + formulas_text,
            "Rules:\n" + "\n".join(f"- {rule}" for rule in rules),
            "Methods:\n" + methods_text,
            "Examples:\n" + examples_text,
        ])

    def get_block(self, block_id):
        # 通过ID 获取blcok 信息
        return self.collection.get(ids=[block_id])

    def search(self, query, n_results=5):
        # query 

        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

    def delete_block(self, block_id):
        self.collection.delete(ids=[block_id])

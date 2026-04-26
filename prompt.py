import json
from typing import Optional

def yield_summarise(group):
    result = f"""
You are converting course PDF pages into structured RAG knowledge blocks.

Return ONLY valid JSON in this exact shape:
{{
  "blocks": [
    {{
      "block_id": "short_stable_id",
      "title": "topic title",
      "summary": "clear summary",
      "key_points": ["point 1", "point 2"],
      "formulas": [
        {{
          "name": "formula name",
          "expression": "formula expression",
          "explanation": "what it means"
        }}
      ],
      "rules": ["rule 1", "rule 2"],
      "methods": [
        {{
          "name": "method name",
          "steps": ["step 1", "step 2"]
        }}
      ],
      "examples": [
        {{
          "question": "example question",
          "answer": "example answer"
        }}
      ],
      "source_pages": "{group['source_pages']}",
      "filename": "{group['filename']}"
    }}
  ]
}}

Strict JSON rules:
- Return ONLY JSON.
- Do not wrap in markdown.
- Do not explain anything before or after the JSON.
- Do not use Python dict syntax.
- Use double quotes for all JSON keys and string values.
- Do not put raw newline characters inside JSON strings. Use spaces instead.
- Escape any required newline inside strings as \\n.
- Do not use trailing commas.
- The top-level key must be exactly "blocks".
- "blocks" must be a non-empty array.
- Every block must include: block_id, title, summary, key_points, formulas, rules, methods, examples, source_pages, filename.
- key_points, formulas, rules, methods, and examples must always be arrays, even if empty.
- formulas must be an array of objects with name, expression, and explanation.
- methods must be an array of objects with name and steps.
- examples must be an array of objects with question and answer.
- If information is missing, use an empty string or empty array.
- Rewrite corrupted formula symbols into simple ASCII text when possible.

Formula and symbol rules:
- Rewrite formulas in simple ASCII/plain text whenever possible.
- Do not copy corrupted PDF symbols.
- Prefer readable ASCII names like sum, product, alpha_i, h_i, s_t, P(y_t | y_<t, x).
- If a formula is unreadable, leave "expression" as an empty string and explain the idea in "explanation".
- Do not include LaTeX commands unless they are simple and valid ASCII.
- Keep formula strings on one line.

PDF content:
{group["text"]}
"""
    return result


def _repair_llm_json(client, model, broken_content, error):
    repair_prompt = f"""
You returned invalid JSON.

Fix the content below into ONLY valid JSON with exactly this shape:
{{
  "blocks": [
    {{
      "block_id": "short_stable_id",
      "title": "topic title",
      "summary": "clear summary",
      "key_points": ["point 1"],
      "formulas": [
        {{
          "name": "formula name",
          "expression": "formula expression",
          "explanation": "what it means"
        }}
      ],
      "rules": ["rule 1"],
      "methods": [
        {{
          "name": "method name",
          "steps": ["step 1"]
        }}
      ],
      "examples": [
        {{
          "question": "example question",
          "answer": "example answer"
        }}
      ],
      "source_pages": "source pages",
      "filename": "filename.pdf"
    }}
  ]
}}

Rules:
- Return ONLY JSON.
- Do not wrap in markdown.
- Do not explain anything.
- Do not use Python dict syntax.
- Keep the same information if possible.
- Escape all newlines inside JSON strings as \\n.
- If a field is unknown, use an empty string or empty list.
- The value of "blocks" must be a JSON array, not a string.

Original parse error:
{error}

Broken content:
{broken_content}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": repair_prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    return response.choices[0].message.content


def rewrite(question, chat_history=None):
    if chat_history is None:
        normalized_chat_history = []
    elif isinstance(chat_history, list):
        normalized_chat_history = [str(item) for item in chat_history]
    elif isinstance(chat_history, str):
        normalized_chat_history = [chat_history] if chat_history.strip() else []
    else:
        normalized_chat_history = [str(chat_history)]

    chat_history_json = json.dumps(
        normalized_chat_history,
        ensure_ascii=False,
        indent=2,
    )

    ans = f"""
You are preparing a user question for RAG retrieval.

Your task:
1. Rewrite the user's latest question into a clear search query for retrieval.
2. Extract a short list of useful keywords.
3. Classify the query intent.
4. Preserve the chat history as a JSON array of strings.

User question:
{question}

Chat history:
{chat_history_json}

Return ONLY valid JSON in exactly this shape:
{{
  "rewritten_query": "clear standalone retrieval query",
  "keywords": ["keyword1", "keyword2"],
  "query_type": "summary | meaning | lookup | comparison | explanation | preference | procedure | open",
  "chat_history": ["previous message 1", "previous message 2"]
}}

Rules:
- Return ONLY JSON.
- Do not wrap the JSON in markdown.
- Do not explain anything before or after the JSON.
- Use double quotes for all keys and string values.
- "rewritten_query" must be a concise standalone search query.
- "keywords" must be a JSON array of short strings.
- Include 3 to 8 keywords when possible.
- "query_type" must be exactly one of:
  "summary", "meaning", "lookup", "comparison", "explanation", "preference", "procedure", "open".
- "chat_history" must remain a JSON array of strings.
- Keep the user's original language unless rewriting into clearer wording is necessary.
- Remove filler words, but keep the actual information need.
- If the user asks about a specific concept, include that concept explicitly in "rewritten_query".
- If there is not enough context, make the best reasonable rewrite from the latest user question.
"""

    return ans

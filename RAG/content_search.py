import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..llm import model_init
except ImportError:
    from llm import model_init

try:
    from ..prompt import rewrite
except ImportError:
    from prompt import rewrite


def rewrite_question(question: str, chat_history=None):
    client, model = model_init()
    prompt = rewrite(question, chat_history or [])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_content = response.choices[0].message.content
    parsed_content = json.loads(raw_content)
    print(json.dumps(parsed_content, ensure_ascii=False, indent=2))
    return parsed_content


if __name__ == "__main__":
    rewrite_question(
        question="黑袍纠察队第5季的第三集中，祖国人遭遇了什么打击？",
        chat_history=[],
    )

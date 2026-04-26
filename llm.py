from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

def model_init():
    env_path = Path(__file__).resolve().with_name(".env")
    load_dotenv(env_path)

    # model detail 
    model = os.getenv("MODELSCOPE_MODEL_ID")
    if not model:
        raise ValueError("missing MODELSCOPE_MODEL_ID")

    api_key = os.getenv("MODELSCOPE_SDK_TOKEN")
    base_url = os.getenv("MODELSCOPE_BASE_URL")

    if not api_key:
        raise ValueError("missing MODELSCOPE_SDK_TOKEN")

    if not base_url:
        raise ValueError("missing MODELSCOPE_BASE_URL")

    client = OpenAI(
        api_key=os.getenv("MODELSCOPE_SDK_TOKEN"),
        base_url=os.getenv("MODELSCOPE_BASE_URL"),
    )

    return client, model

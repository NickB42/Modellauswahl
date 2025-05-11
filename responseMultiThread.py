import time
import json
import random
from pathlib import Path
import logging
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm
from requests.exceptions import HTTPError

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI
from gen_ai_hub.proxy.native.amazon.clients import Session
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

MAX_TOKENS = 400
DEFAULT_RUNS = 50
INITIAL_BACKOFF = 1
MAX_BACKOFF = 30
RESULTS_DIR = Path("results2")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Model case factories
def simple_case(query: str, model_name: str):
    return {
        "data": {"deployment_id": model_name, "messages": query},
        "process": lambda resp, elapsed: [model_name, resp.usage_metadata["output_tokens"], elapsed],
    }

def anthropic_case(query: str, model_name: str):
    return {
        "data": {"deployment_id": model_name, "messages": [{"role": "user", "content": [{"text": query}]}]},
        "process": lambda resp, elapsed: [model_name, resp['usage']['outputTokens'], elapsed],
    }

def ibm_case(query: str, model_name: str):
    return {
        "data": {"deployment_id": model_name, "messages": query},
        "process": lambda resp, elapsed: [model_name, resp["results"][0]["tokenCount"], elapsed],
    }

model_factories = {
    "gpt": simple_case,
    "gemini": simple_case,
    "anthropic": anthropic_case,
    "ibm--granite-13b-chat": ibm_case,
}


def _send_request(payload: dict, model_name: str):
    """
    Underlying request logic; returns (response, elapsed_seconds).
    """
    start_time = time.time()

    if "amazon" in model_name or "anthropic" in model_name:
        client = Session().client(model_name=model_name)
        conversation = payload.get("messages")
        response = client.converse(
            messages=conversation,
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )
    elif model_name == "ibm--granite-13b-chat":
        client = Session().client(model_name="amazon--titan-text-express")
        body = json.dumps({
            "inputText": payload.get("messages"),
            "textGenerationConfig": {"maxTokenCount": MAX_TOKENS},
        })
        raw = client.invoke_model(body=body)
        response = json.loads(raw.get("body").read())
    else:
        proxy_client = get_proxy_client('gen-ai-hub')
        prompt = payload.get("messages")
        if model_name.startswith("gpt") or model_name in ("o1", "o3-mini", "mistralai--mistral-large-instruct"):
            llm = ChatOpenAI(proxy_model_name=model_name, proxy_client=proxy_client, max_tokens=MAX_TOKENS)
        else:
            llm = ChatVertexAI(proxy_model_name=model_name, proxy_client=proxy_client, max_tokens=MAX_TOKENS)
        response = llm.invoke(prompt)

    elapsed = time.time() - start_time
    return response, elapsed


def make_request(payload: dict, model_name: str):
    """
    Wraps _send_request() with retry on 429 using exponential backoff + jitter.
    Never gives up—will keep retrying until success or non-429 error.
    """
    backoff = INITIAL_BACKOFF
    while True:
        try:
            response, elapsed = _send_request(payload, model_name)
            time.sleep(5)
            return response, elapsed
        except HTTPError as e:
            print(e)
            logger.error(f"fgrober fehle2r: {e}")
            if "anthropic" in model_name:
                status = getattr(e.response, 'status_code', None)
            else:
                status = getattr(e.response, 'status_code', None)
            if status == 429:
                logger.info("ja rate limit, doof ne")
                # Honor server's Retry-After if present
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    sleep_time = float(retry_after)
                else:
                    # exponential backoff with jitter
                    sleep_time = min(backoff, MAX_BACKOFF) * (0.5 + random.random() * 0.5)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                logger.warning(f"Rate limit hit for {model_name}, sleeping {sleep_time:.1f}s before retrying.")
                time.sleep(sleep_time)
                continue
            # non-429 errors bubble up
            print(e)


def process_single(model_name: str, query: str, run_index: int):
    """
    Perform a single request to a model and return [model_name, output_tokens, elapsed_seconds].
    """
    # Choose the right factory
    if "gpt" in model_name or model_name in ("o1", "o3-mini") or "mistralai" in model_name:
        key = "gpt"
    elif "gemini" in model_name:
        key = "gemini"
    elif "anthropic" in model_name or "amazon" in model_name:
        key = "anthropic"
    else:
        key = model_name

    factory = model_factories[key]
    case_info = factory(query, model_name)
    response, elapsed = make_request(case_info["data"], model_name)
    return case_info["process"](response, elapsed)


def write_json(path: Path, data):
    """Write data to JSON file with pretty formatting and log the result count."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    logger.info(f"Wrote {len(data)} entries to {path}")

def main():
    models = [
        "mistralai--mistral-large-instruct",
        "anthropic--claude-3-haiku", "anthropic--claude-3-opus", "anthropic--claude-3.7-sonnet",
        "gpt-4o", "gpt-4o-mini", "o1", "o3-mini",
        "gemini-2.0-flash", "gemini-1.5-pro",
    ]
    questions = [
        "What is the capital of Australia?",
        "What is the distance between the Earth and the Moon?",
        "Who was the first woman to win a Nobel Prize?",
        "What is the law of conservation of energy?",
        "How does a combustion engine work?",
        "What is the Pythagorean theorem?",
        "What is a utility analysis?",
        "What are the key principles of quantum mechanics?",
        "What is the BLEU score in Machine Learning?",
        "What is the F-1 score in Machine Learning?",
    ]

    executors = {
        model: concurrent.futures.ThreadPoolExecutor(max_workers=1)
        for model in models
    }

    futures = {}
    for model in models:
        for question in questions:
            for run_index in range(DEFAULT_RUNS):
                future = executors[model].submit(process_single, model, question, run_index)
                futures[future] = (model, question, run_index)

    results = []
    for fut in tqdm(concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Overall progress",
                    unit="task"):
        m, q, i = futures[fut]
        try:
            results.append(fut.result())
        except Exception as err:
            logger.error(f"Error in {m} run {i}, question '{q}': {err}")

    for executor in executors.values():
        executor.shutdown(wait=False)

    efficiency = [
        [model, tokens, elapsed, round(elapsed / tokens, 5)]
        for model, tokens, elapsed in results if tokens
    ]

    grouped = defaultdict(list)
    for record in efficiency:
        grouped[record[0]].append(record)

    overall_path = RESULTS_DIR / "results.json"
    write_json(overall_path, efficiency)

    for model_name, entries in grouped.items():
        safe = model_name.replace("/", "_")
        file_path = RESULTS_DIR / f"{safe}_results.json"
        write_json(file_path, entries)

if __name__ == "__main__":
    main()
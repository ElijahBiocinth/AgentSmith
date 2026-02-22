"""
Multi-model review tool — multi-provider (OpenAI-compatible).

Best used for critic/consensus where tool-calling is NOT required.
Primary agent should remain OpenAI for stable tool integration.
"""
import os
import json
import asyncio
import logging
import httpx

from ouroboros.utils import utc_now_iso
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.providers import parse_model_id, get_provider

log = logging.getLogger(__name__)

MAX_MODELS = 10
CONCURRENCY_LIMIT = 5


def get_tools():
    return [
        ToolEntry(
            name="multi_model_review",
            schema={
                "name": "multi_model_review",
                "description": (
                    "Send code/text to multiple models (multi-provider) for independent review. "
                    "Models can be plain 'gpt-5.2' or prefixed 'deepseek:deepseek-chat', "
                    "'qwen:qwen-max', 'hf:org/model'. Returns structured verdicts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "prompt": {"type": "string"},
                        "models": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content", "prompt", "models"],
                },
            },
            handler=_handle_multi_model_review,
        )
    ]


def _handle_multi_model_review(ctx: ToolContext, content: str = "", prompt: str = "", models: list = None) -> str:
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, _multi_model_review_async(content, prompt, models, ctx)).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


async def _query_model(client, model_id, messages, semaphore):
    async with semaphore:
        prov, mdl = parse_model_id(model_id)
        try:
            base_url, api_key = get_provider(prov)
            url = base_url.rstrip("/") + "/chat/completions"
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": mdl, "messages": messages, "temperature": 0.2},
                timeout=120.0,
            )
            if resp.status_code != 200:
                t = resp.text
                err = t[:200] + (" [truncated]" if len(t) > 200 else "")
                return model_id, f"HTTP {resp.status_code}: {err}", None
            return model_id, resp.json(), dict(resp.headers)
        except asyncio.TimeoutError:
            return model_id, "Error: Timeout after 120s", None
        except Exception as e:
            s = str(e)
            return model_id, f"Error: {s[:200] + (' [truncated]' if len(s) > 200 else '')}", None


async def _multi_model_review_async(content: str, prompt: str, models: list, ctx: ToolContext):
    if not content:
        return {"error": "content is required"}
    if not prompt:
        return {"error": "prompt is required"}
    if not models:
        return {"error": "models list is required"}
    if len(models) > MAX_MODELS:
        return {"error": f"Too many models requested ({len(models)}). Maximum is {MAX_MODELS}."}

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return {"error": "OPENAI_API_KEY not set (required by system)."}

    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with httpx.AsyncClient() as client:
        tasks = [_query_model(client, m, messages, semaphore) for m in models]
        results = await asyncio.gather(*tasks)

    review_results = []
    for model_id, result, headers_dict in results:
        review_result = _parse_model_response(model_id, result)
        _emit_usage_event(review_result, ctx)
        review_results.append(review_result)

    return {"model_count": len(models), "results": review_results}


def _parse_model_response(model_id: str, result) -> dict:
    if isinstance(result, str):
        return {"model": model_id, "verdict": "ERROR", "text": result, "tokens_in": 0, "tokens_out": 0}

    try:
        choices = result.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
    except Exception:
        text = ""

    verdict = "UNKNOWN"
    for line in str(text).split("\n")[:3]:
        u = line.upper()
        if "PASS" in u:
            verdict = "PASS"
            break
        if "FAIL" in u:
            verdict = "FAIL"
            break

    usage = result.get("usage", {}) or {}
    return {
        "model": model_id,
        "verdict": verdict,
        "text": text,
        "tokens_in": int(usage.get("prompt_tokens", 0) or 0),
        "tokens_out": int(usage.get("completion_tokens", 0) or 0),
    }


def _emit_usage_event(review_result: dict, ctx: ToolContext) -> None:
    if ctx is None:
        return
    usage_event = {
        "type": "llm_usage",
        "ts": utc_now_iso(),
        "task_id": getattr(ctx, "task_id", "") or "",
        "usage": {
            "prompt_tokens": int(review_result.get("tokens_in", 0) or 0),
            "completion_tokens": int(review_result.get("tokens_out", 0) or 0),
            "cost": 0.0,
        },
        "category": "review",
    }
    if getattr(ctx, "event_queue", None) is not None:
        try:
            ctx.event_queue.put_nowait(usage_event)
            return
        except Exception:
            pass
    if hasattr(ctx, "pending_events"):
        ctx.pending_events.append(usage_event)

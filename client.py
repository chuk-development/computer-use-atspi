"""Model client for the computer-use demo.

OpenAI-compatible chat/completions over plain HTTP, so the SAME loop drives any
provider that speaks that shape. Default is Deep Infra (GLM). Swap provider by
setting CU_BASE_URL / CU_MODEL / CU_API_KEY.

Examples:
  Deep Infra GLM (default):
    CU_BASE_URL=https://api.deepinfra.com/v1/openai
    CU_MODEL=zai-org/GLM-4.5V
  OpenAI:
    CU_BASE_URL=https://api.openai.com/v1
    CU_MODEL=gpt-4o           # or a computer-use capable model
"""
from __future__ import annotations

import json
import os

import requests


def _load_dotenv() -> None:
    """Populate os.environ from a sibling .env file (no dependency needed)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


_load_dotenv()

BASE_URL = os.environ.get("CU_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
MODEL = os.environ.get("CU_MODEL", "zai-org/GLM-5.3-Flash")
REASONING_EFFORT = os.environ.get("CU_REASONING_EFFORT", "high")  # high|medium|low|off
# NOTE: with tools present (always, for computer-use), GLM-5.3-Flash only emits
# reasoning_content at "high". "medium"/"low" return no visible thinking here.
API_KEY = (
    os.environ.get("CU_API_KEY")
    or os.environ.get("DEEPINFRA_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)


def chat(messages: list, tools: list, temperature: float = 0.2, max_tokens: int = 1024) -> dict:
    if not API_KEY:
        raise RuntimeError("No API key: set CU_API_KEY (or DEEPINFRA_API_KEY / OPENAI_API_KEY).")
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if REASONING_EFFORT and REASONING_EFFORT.lower() != "off":
        payload["reasoning_effort"] = REASONING_EFFORT  # turns on GLM thinking (reasoning_content)
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"model API {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return data["choices"][0]["message"], data.get("usage", {}) or {}


def chat_stream(messages, tools, on_delta=None, temperature: float = 0.2, max_tokens: int = 1024):
    """Streamed chat. Calls on_delta(kind, text) as reasoning/content arrives
    (kind is 'reasoning' or 'content'). Returns (assembled_message, usage) at the
    end. Tool-call fragments are reassembled by index."""
    if not API_KEY:
        raise RuntimeError("No API key: set CU_API_KEY (or DEEPINFRA_API_KEY / OPENAI_API_KEY).")
    payload = {
        "model": MODEL, "messages": messages, "tools": tools, "tool_choice": "auto",
        "temperature": temperature, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if REASONING_EFFORT and REASONING_EFFORT.lower() != "off":
        payload["reasoning_effort"] = REASONING_EFFORT
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload, stream=True, timeout=180,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"model API {resp.status_code}: {resp.text[:500]}")

    reasoning, content, usage = [], [], {}
    tcs: dict = {}
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices", []) or []:
            delta = choice.get("delta", {}) or {}
            rc = delta.get("reasoning_content")
            if rc:
                reasoning.append(rc)
                if on_delta:
                    on_delta("reasoning", rc)
            c = delta.get("content")
            if c:
                content.append(c)
                if on_delta:
                    on_delta("content", c)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tcs.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
    calls = [
        {"id": v["id"], "type": "function",
         "function": {"name": v["name"], "arguments": v["args"]}}
        for _, v in sorted(tcs.items())
        if v.get("name")
    ]
    msg = {"role": "assistant", "content": "".join(content) or None,
           "reasoning_content": "".join(reasoning), "tool_calls": calls or None}
    return msg, usage

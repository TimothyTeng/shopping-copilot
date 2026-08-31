"""Minimal OpenAI-compatible chat client, standard library only.

Deliberately not a dependency. `requirements.txt` stays empty on the scored path,
and a network-disabled environment has nothing extra to fail on — this module is
only imported by code that has already decided to talk to a model.

Shared by the runtime backend (`hyde.py`) and the offline tooling
(`tools/genprobes.py`) so there is exactly one place that knows the wire format.
"""
from __future__ import annotations

import json
import urllib.request

DEFAULT_BASE = "http://localhost:30800/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def chat(url: str, model: str, messages: list[dict], max_tokens: int,
         timeout: float, key: str | None = None,
         temperature: float = 0.0, seed: int = 0,
         json_object: bool = False) -> tuple[str, int, int]:
    """One completion. Returns (text, prompt_tokens, completion_tokens).

    Raises on any transport or protocol failure — callers decide what a failure
    means. At runtime that is "keep the Tier-0 ranking"; offline it is "skip this
    sample". Defaults are deterministic (temperature 0, fixed seed) so that a
    measurement is reproducible and a response cache is sound.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"] or ""
    usage = body.get("usage") or {}
    return (text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)))

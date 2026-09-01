"""
Chat-completion clients: OpenRouter (hosted, paid) and Ollama (local, free).

DEVIATION FROM THE PAPER: the paper calls the Gemini API directly. Per the
user's explicit choices (no Gemini access; now running on an HPC cluster,
both model tiers run locally and for free via Ollama -- see
NOTES_AND_ASSUMPTIONS.md section (f)), this codebase can run either
backend for either tier -- see config.py for the model/backend mapping and
its caveats.

Retry behaviour mirrors the paper's stated reasoning (Section 5.2): "the
small degree of randomness [temperature=0.1] allows for agents to repeat
queries which returned an incorrectly-formatted output from the LLM, and to
deliver results in the correct format in subsequent iterations." We
implement that literally: on a malformed/unparseable response, re-query
(same prompt, same non-zero temperature) up to MAX_RETRIES times. This
retry/parsing logic is shared by both backends via BaseChatClient, so
switching backends never changes behaviour around malformed output.
"""
import os
import re

import requests

# Ollama spawns as many threads as the node's full logical CPU count by
# default (llama.cpp's nproc-based auto-detect), which on a SLURM cluster
# routinely exceeds the actual cgroup-limited core count for the job --
# measured impact on explorer.northeastern.edu: 128 threads fighting over a
# 48-core cgroup allocation dropped 8B CPU throughput to 0.02 tok/s (barrier
# contention in llama.cpp's per-layer thread sync). Pinning num_thread to the
# job's real allocation fixed it (12.34 tok/s on an AVX512 Cascade Lake
# node). SLURM_CPUS_PER_TASK is unset outside a job (e.g. local dev), so
# fall back to os.cpu_count().
_NUM_THREAD = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

from .config import (
    FIXED_PARAMS,
    OLLAMA_BASE_URL,
    OPENROUTER_API_KEY_ENV_VAR,
    OPENROUTER_BASE_URL,
)

MAX_RETRIES = 3
ORDER_PATTERN = re.compile(r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]")

# FLAGGED JUDGMENT CALL / NEW ADAPTATION (not in the paper, disclosed here
# and in NOTES_AND_ASSUMPTIONS.md): Llama 3.1 (unlike the paper's Gemini
# models) tends to emit step-by-step markdown reasoning by default, which
# blows through the paper's fixed 90-token output budget before ever
# reaching a bracketed answer. Rather than raise max_output_tokens (a fixed,
# stated parameter we don't want to silently change), we add a system
# message enforcing terse output for the strict-format calls only. This is
# the same kind of per-model prompt adaptation the paper itself describes
# doing for Gemini Pro vs Flash (Appendix 3), applied here to a different
# model family that needs a different (stronger) fix for a different
# failure mode.
_STRICT_FORMAT_SYSTEM_MESSAGE = (
    "You are a supply chain ordering agent. You must respond with ONLY the "
    "final answer in the exact requested format. Never show step-by-step "
    "reasoning, headers, or any explanation text."
)


class MalformedOutputError(Exception):
    pass


class BaseChatClient:
    """
    Shared retry/parsing logic. Subclasses only need to implement
    `_call_raw(messages) -> str`.
    """

    def _call_raw(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def get_order_decision(self, prompt: str, max_order: int = FIXED_PARAMS["max_order_amount"]) -> int:
        """
        Send `prompt` as a user message (with a strict-format system
        message prepended), parse an integer order amount out of a
        `[[N]]`-formatted reply, retrying on malformed output up to
        MAX_RETRIES times (see module docstring).
        """
        messages = [
            {"role": "system", "content": _STRICT_FORMAT_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content = self._call_raw(messages)
                match = ORDER_PATTERN.search(content)
                if not match:
                    raise MalformedOutputError(f"No [[N]] pattern found in: {content!r}")
                value = float(match.group(1))
                # Hard-coded backstop, independent of prompt instructions
                # (Section 7.1: "constraints are handled ... through
                # hard-coded backstops in the environment").
                return int(max(0, min(round(value), max_order)))
            except (MalformedOutputError, KeyError, ValueError, requests.RequestException) as e:
                last_error = e
        raise MalformedOutputError(
            f"Failed to get a well-formatted order decision after {MAX_RETRIES} attempts: {last_error}"
        )

    def chat(self, prompt: str, strict_format: bool = False) -> str:
        """
        Free-form chat turn, used during negotiation exchanges. Pass
        strict_format=True only when the reply must itself contain a
        parseable [[N]] answer (e.g. the negotiation final-answer turn) --
        mid-negotiation turns should stay conversational.
        """
        messages = []
        if strict_format:
            messages.append({"role": "system", "content": _STRICT_FORMAT_SYSTEM_MESSAGE})
        messages.append({"role": "user", "content": prompt})
        return self._call_raw(messages)


class LLMClient(BaseChatClient):
    """OpenRouter-backed client (hosted, paid per token)."""

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
        if not self.api_key:
            raise RuntimeError(
                f"No API key found. Set the {OPENROUTER_API_KEY_ENV_VAR} environment "
                "variable, or pass api_key= explicitly."
            )

    def _call_raw(self, messages: list[dict]) -> str:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": FIXED_PARAMS["temperature"],
                "max_tokens": FIXED_PARAMS["max_output_tokens"],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class OllamaClient(BaseChatClient):
    """
    Local-inference client via Ollama's REST API (http://localhost:11434).
    Zero marginal cost per call; speed/feasibility depends entirely on local
    hardware -- both tiers run this way now (see NOTES_AND_ASSUMPTIONS.md
    section (f) for the HPC node-selection rules this implies: the 70B/
    "large" tier requires a GPU node, CPU is ~300x too slow for it).
    """

    def __init__(self, model: str):
        self.model = model

    def _call_raw(self, messages: list[dict]) -> str:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": FIXED_PARAMS["temperature"],
                    "num_predict": FIXED_PARAMS["max_output_tokens"],
                    "num_thread": _NUM_THREAD,
                    # Ollama otherwise defaults num_ctx to the model's full
                    # trained context (131072 for Llama 3.1), which on the
                    # 70B/H200 test bloated the KV cache to ~40GB and made
                    # the first prompt eval take 19s for 17 tokens. Our
                    # prompts are short (well under 4096 tokens); capping
                    # this avoids that cold-start cost on every backend.
                    "num_ctx": 4096,
                },
            },
            timeout=600,  # local CPU inference can be slower than a hosted API
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

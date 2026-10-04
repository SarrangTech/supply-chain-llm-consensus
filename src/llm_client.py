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
import sys

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
    EXPERIMENT_SEED,
    FIXED_PARAMS,
    OLLAMA_BASE_URL,
    OPENROUTER_API_KEY_ENV_VAR,
    OPENROUTER_BASE_URL,
)

MAX_RETRIES = 3
ORDER_PATTERN = re.compile(r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]")
_BRACKET_CONTENT_PATTERN = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_order_answer(text: str, max_order: int) -> int | None:
    """
    Extract a final integer order amount from a model reply. Tries the
    strict [[N]] format first (bare number, exactly as instructed). Falls
    back to a lenient extraction if the model showed reasoning inside the
    brackets instead of a bare number (e.g.
    "[[sqrt(9.00*7.00) = 7.94, rounded to 8]]") -- measured to happen more
    often once negotiation.py's final-answer question was grounded in the
    actual transcript (NOTES_AND_ASSUMPTIONS.md section (h)): richer
    context makes the model more likely to imitate the conversation's own
    shown-work style even in a strict-format answer. Takes the LAST number
    inside the brackets, since a model restating a calculation states the
    final/rounded result last ("= 7.94, rounded to 8" -> 8, not 7.94 or the
    9/7 inputs). Returns None if no number could be extracted at all.
    """
    match = ORDER_PATTERN.search(text)
    if match:
        return int(max(0, min(round(float(match.group(1))), max_order)))
    bracket_match = _BRACKET_CONTENT_PATTERN.search(text)
    if bracket_match:
        numbers = _NUMBER_PATTERN.findall(bracket_match.group(1))
        if numbers:
            return int(max(0, min(round(float(numbers[-1])), max_order)))
    return None

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
                # Hard-coded backstop, independent of prompt instructions
                # (Section 7.1: "constraints are handled ... through
                # hard-coded backstops in the environment"). parse_order_answer
                # already clamps to max_order.
                value = parse_order_answer(content, max_order)
                if value is None:
                    raise MalformedOutputError(f"No [[N]] pattern found in: {content!r}")
                return value
            except (MalformedOutputError, KeyError, ValueError, requests.RequestException) as e:
                last_error = e
                # Diagnostic visibility into how often the retry path fires
                # (see NOTES_AND_ASSUMPTIONS.md section (g) -- Llama 3.1 is
                # documented to break the strict [[N]] format more often
                # than the paper's Gemini models; this makes that measurable
                # instead of theorized).
                model = getattr(self, "model", "?")
                print(f"RETRY model={model} attempt={attempt + 1}/{MAX_RETRIES} error={e}", file=sys.stderr, flush=True)
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
                    # RQ3/RQ5 (2026-09-29): threaded from EXPERIMENT_SEED so
                    # repeated reruns can actually vary LLM sampling in a
                    # controlled way instead of relying on unseeded
                    # process-time randomness.
                    "seed": EXPERIMENT_SEED,
                    # Ollama otherwise defaults num_ctx to the model's full
                    # trained context (131072 for Llama 3.1), which on the
                    # 70B/H200 test bloated the KV cache to ~40GB and made
                    # the first prompt eval take 19s for 17 tokens. Our
                    # prompts are short (well under 4096 tokens); capping
                    # this avoids that cold-start cost on every backend.
                    "num_ctx": 4096,
                },
            },
            # 2026-10-01: raised from 600s after 4 consecutive real failures
            # (both Gemma and Qwen, both metrics) on the --include-transcripts
            # small-tier CPU reruns, all timing out on the genuine first real
            # LLM call (step 0's degenerate zero-EOQ case is skipped in code,
            # so this is step 1's first call) at exactly 600s, despite a
            # successful warm-up immediately before each. Not reproduced on
            # the original (non-transcript) small-tier CPU runs from the day
            # before. No code-level cause found connecting --include-transcripts
            # to this (transcript_sink is a pure output accumulator, never fed
            # back into any prompt) -- most likely a cluster load/CPU-
            # performance variation between the two days, not a logic bug.
            timeout=1200,  # local CPU inference can be slower than a hosted API
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

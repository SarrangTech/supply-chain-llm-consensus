"""
Deterministic stand-in for LLMClient, used only to validate that the full
pipeline (environment, tools, prompts, LangGraph frameworks, metrics,
experiment grid) runs end-to-end without making any paid API calls.

This produces PLAUSIBLE-LOOKING but NOT MEANINGFUL results: it is a wiring
test, not a substitute for real experiments. Any numbers produced with this
client must not be reported as replicated results.
"""
import re

from .llm_client import ORDER_PATTERN


class MockLLMClient:
    def __init__(self, model: str = "mock", bias: float = 1.0):
        self.model = model
        self.bias = bias  # lets standalone vs "pro-like" mocks differ slightly, for testing

    def get_order_decision(self, prompt: str, max_order: int = 100) -> int:
        # Try to imitate "follow the tool output" behaviour: if a tool value
        # is mentioned in the prompt, echo it (perturbed slightly); otherwise
        # fall back to a naive heuristic based on the last observed demand
        # mentioned in the prompt.
        tool_match = re.search(r"result obtained from this tool is:\s*\[?(\d+(?:\.\d+)?)\]?", prompt, re.IGNORECASE)
        if tool_match:
            value = float(tool_match.group(1)) * self.bias
        else:
            demand_match = re.search(r"Last 30 steps showed a demand of \[([^\]]*)\]", prompt)
            if demand_match:
                nums = [float(x) for x in demand_match.group(1).split(",") if x.strip()]
                value = (sum(nums) / len(nums)) if nums else 1.0
            else:
                value = 1.0
        return int(max(0, min(round(value), max_order)))

    def chat(self, prompt: str, strict_format: bool = False) -> str:
        decision = self.get_order_decision(prompt)
        return f"I propose an order of {decision} units. [[{decision}]]"

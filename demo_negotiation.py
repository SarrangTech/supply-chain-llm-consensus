#!/usr/bin/env python
"""
Live demo: watch two agents actually negotiate an order quantity, turn by
turn, using the real negotiation graph and prompts from
src/frameworks/negotiation.py -- not a scripted mockup. Talks to whatever
Ollama server is reachable at OLLAMA_BASE_URL (or OLLAMA_HOST), same as the
real experiment runner.

Usage:
    python demo_negotiation.py [small|large]   # default: large (70B)
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/demo_negotiation.py", 1)[0])

from src.config import FIXED_PARAMS, OLLAMA_MODELS
from src.llm_client import OllamaClient
from src.tools import eoq_tool
from src.frameworks.negotiation import _NEGOTIATION_GRAPH

tier = sys.argv[1] if len(sys.argv) > 1 else "large"
model = OLLAMA_MODELS[tier]
print(f"=== Live negotiation demo -- model: {model} ({tier} tier) ===\n", flush=True)

# Two synthetic agents with different recent demand -- gives them a genuine
# reason to disagree on order quantity (retailer sees choppier demand than
# the distributor further upstream), same shape of input the real
# simulation feeds into this same code path.
downstream_demand = [8, 12, 6, 15, 9, 11, 7, 14, 10, 13] * 3   # retailer: noisy
upstream_demand = [10, 10, 11, 10, 9, 10, 10, 11, 10, 10] * 3  # distributor: smoothed

downstream_eoq = eoq_tool(downstream_demand, FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], FIXED_PARAMS["demand_forecast_lookback"])
upstream_eoq = eoq_tool(upstream_demand, FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], FIXED_PARAMS["demand_forecast_lookback"])
print(f"Downstream agent's computed EOQ: {downstream_eoq:.2f}")
print(f"Upstream agent's computed EOQ:   {upstream_eoq:.2f}\n", flush=True)

init_state = {
    "downstream_idx": 0,
    "upstream_idx": 1,
    "downstream_client": OllamaClient(model=model),
    "upstream_client": OllamaClient(model=model),
    "downstream_eoq": downstream_eoq,
    "upstream_eoq": upstream_eoq,
    "max_order": FIXED_PARAMS["max_order_amount"],
}

printed = 0
for step_state in _NEGOTIATION_GRAPH.stream(init_state, stream_mode="values"):
    transcript = step_state.get("transcript", [])
    for speaker, text in transcript[printed:]:
        label = "RETAILER (downstream)" if speaker == "downstream" else "DISTRIBUTOR (upstream)"
        print(f"[{label}]: {text}\n", flush=True)
        printed += 1
        time.sleep(0.3)

print("=== Final decision ===")
print(f"Downstream (retailer) final order:    {step_state['downstream_order']}")
print(f"Upstream (distributor) final order:   {step_state['upstream_order']}")

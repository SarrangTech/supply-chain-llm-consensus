#!/usr/bin/env python
"""
Live demo variant #2: same real model, same real OllamaClient, but with
custom prompts that ground the negotiation in actual inventory-management
state (on-hand inventory, backlog, recent demand, lead time) instead of
just an abstract EOQ number.

NOTE: this is a demo-only prompt variant for illustration purposes. The
paper-faithful negotiation prompts actually used in the committed 200-step
results (src/frameworks/negotiation.py, src/prompts.py) are intentionally
minimal -- they only ever reference the EOQ bound, per the paper's own
Figure 9 transcript. This script does NOT change that code or those
results; it's a separate, clearly-labeled illustration of what the same
model does when given richer operational context.

Usage:
    python demo_negotiation_grounded.py [small|large]   # default: large (70B)
"""
import re
import sys

sys.path.insert(0, __file__.rsplit("/demo_negotiation_grounded.py", 1)[0])

from src.config import FIXED_PARAMS, OLLAMA_MODELS
from src.llm_client import OllamaClient
from src.tools import eoq_tool

tier = sys.argv[1] if len(sys.argv) > 1 else "large"
model = OLLAMA_MODELS[tier]
NUM_ITER = FIXED_PARAMS["num_negotiation_iters"]
ORDER_PATTERN = re.compile(r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]")

print(f"=== Grounded inventory-management negotiation demo -- model: {model} ({tier} tier) ===\n", flush=True)

retailer = {
    "role": "retailer",
    "inventory": 12,
    "backlog": 3,
    "demand_history": [18, 22, 15, 25, 20],
    "lead_time": FIXED_PARAMS["lead_time"],
}
distributor = {
    "role": "distributor",
    "inventory": 40,
    "backlog": 0,
    "demand_history": [20, 19, 21, 20, 20],
    "lead_time": FIXED_PARAMS["lead_time"],
}
for agent in (retailer, distributor):
    agent["eoq"] = eoq_tool(agent["demand_history"], FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], lookback=5)

for agent in (retailer, distributor):
    print(f"{agent['role'].upper()}: inventory={agent['inventory']}, backlog={agent['backlog']}, "
          f"recent demand={agent['demand_history']}, lead_time={agent['lead_time']}, EOQ estimate={agent['eoq']:.1f}")
print(flush=True)


def situation_line(agent):
    return (
        f"You are the {agent['role']} in a 3-tier supply chain. Your current on-hand inventory "
        f"is {agent['inventory']} units, with a backlog of {agent['backlog']} unfulfilled units. "
        f"Over the last {len(agent['demand_history'])} periods, demand you received was "
        f"{agent['demand_history']}. It takes {agent['lead_time']} periods for a new order to "
        f"arrive. Based on this, an EOQ calculation suggests ordering around {agent['eoq']:.1f} "
        f"units, but you should weigh your actual inventory situation, not just this number."
    )


def intro_prompt(agent):
    return (
        situation_line(agent) + " You are about to negotiate your order quantity with your "
        "supply chain partner, whose exact inventory situation you don't know. Start the "
        "conversation: explain your current situation in concrete terms (inventory, backlog, "
        "recent demand) and what you think you need to order, and why."
    )


def turn_prompt(agent, last_msg):
    return (
        situation_line(agent) + f' Your negotiating partner just said: "{last_msg}" '
        "Respond, referencing your actual inventory situation (not just an abstract average), "
        "and propose or agree on a concrete order quantity that balances both sides' operational "
        "needs."
    )


def final_prompt(agent):
    return (
        situation_line(agent) + " Given the conversation so far, what is your final order "
        f"quantity? Respond with ONLY an integer from 0 to {FIXED_PARAMS['max_order_amount']} "
        "enclosed in double brackets, e.g. [[12]]. No explanation."
    )


retailer_client = OllamaClient(model=model)
distributor_client = OllamaClient(model=model)

transcript = []

opening = retailer_client.chat(intro_prompt(retailer))
transcript.append(("RETAILER", opening))

speaker, other = "DISTRIBUTOR", "RETAILER"
current_agent, current_client = distributor, distributor_client
for i in range(2 * NUM_ITER - 1):
    last_msg = transcript[-1][1]
    reply = current_client.chat(turn_prompt(current_agent, last_msg))
    transcript.append((speaker, reply))
    speaker, other = other, speaker
    current_agent, current_client = (retailer, retailer_client) if current_agent is distributor else (distributor, distributor_client)

for label, text in transcript:
    print(f"[{label}]: {text}\n", flush=True)


def final_order(client, agent):
    reply = client.chat(final_prompt(agent), strict_format=True)
    match = ORDER_PATTERN.search(reply)
    return int(max(0, min(round(float(match.group(1))), FIXED_PARAMS["max_order_amount"]))) if match else None


print("=== Final decision ===")
print(f"Retailer final order:     {final_order(retailer_client, retailer)}")
print(f"Distributor final order:  {final_order(distributor_client, distributor)}")

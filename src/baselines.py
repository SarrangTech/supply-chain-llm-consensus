"""
Non-LLM baselines, Section 5.1 and Table 2's baseline rows.

1. (S,s) restocking policy -- "soft"/"weak" baseline. S=100, s=60.
   FLAGGED JUDGMENT CALL: the paper describes this only as "reorders to a
   maximum level (S) when inventory falls below a threshold (s)" without
   specifying whether "inventory" means raw on-hand inventory or inventory
   position (on-hand - backlog + pipeline). We use raw on-hand inventory,
   matching the literal wording.

2. Tool-only baseline -- "hard"/"strong" baseline. The relevant tool's raw
   output IS the order decision, with no LLM in the loop at all (Section 5.1).

3. Chen et al. (2000) baseline (bullwhip experiments only) -- centralised
   demand, moving-average forecast + order-up-to policy.
   FLAGGED JUDGMENT CALL: the paper describes this baseline only at the
   level of "quantifies the bullwhip effect under a moving average forecast
   and order-up-to policy" and "allows for demand centralisation by sharing
   the retailer's demand with all upstream agents." It does not give the
   exact order-up-to formula used. We implement the standard textbook
   version: every agent forecasts using a moving average of the RETAILER's
   (centralised) demand over the same 30-period lookback used elsewhere in
   this codebase, sets an order-up-to level of forecast_mean * (lead_time+1)
   (no explicit safety-stock term, since none is specified), and orders the
   shortfall against its own inventory position.
"""
from .config import FIXED_PARAMS, SS_POLICY
from .tools import demand_forecast_tool, eoq_tool


def ss_policy_decision(agent_obs: dict) -> int:
    """(S, s) policy using raw on-hand inventory."""
    S, s = SS_POLICY["S"], SS_POLICY["s"]
    inv = agent_obs["inventory"]
    if inv < s:
        return max(0, min(S - inv, FIXED_PARAMS["max_order_amount"]))
    return 0


def demand_forecast_tool_only_decision(agent_obs: dict) -> int:
    """
    Strong baseline for cost-minimisation experiments. Expects
    `agent_obs["downstream_demand_history"]` to hold the FULL demand history
    (not just the last-10 slice used for LLM prompts), since the tool looks
    back up to 30 periods.
    """
    forecast = demand_forecast_tool(
        agent_obs["downstream_demand_history"],
        lookback=FIXED_PARAMS["demand_forecast_lookback"],
    )
    return int(max(0, min(round(forecast), FIXED_PARAMS["max_order_amount"])))


def eoq_tool_only_decision(agent_obs: dict) -> int:
    """Strong/hard baseline for bullwhip-effect experiments."""
    eoq = eoq_tool(
        agent_obs["downstream_demand_history"],
        ordering_cost=FIXED_PARAMS["ordering_cost"],
        holding_cost=FIXED_PARAMS["inventory_cost"],
        lookback=FIXED_PARAMS["demand_forecast_lookback"],
    )
    return int(max(0, min(round(eoq), FIXED_PARAMS["max_order_amount"])))


def chen_et_al_2000_decision(
    agent_idx: int,
    agents_obs: list[dict],
    retailer_demand_history: list[int],
    lookback: int = FIXED_PARAMS["demand_forecast_lookback"],
    lead_time: int = FIXED_PARAMS["lead_time"],
) -> int:
    """
    Centralised-demand moving-average order-up-to baseline (Chen et al. 2000),
    per the flagged simplification documented in the module docstring.
    """
    window = retailer_demand_history[-lookback:] if retailer_demand_history else [0]
    forecast_mean = sum(window) / len(window)
    order_up_to = forecast_mean * (lead_time + 1)

    obs = agents_obs[agent_idx]
    pipeline = sum(obs.get("upcoming_shipments", []))
    inventory_position = obs["inventory"] - obs["backlog"] + pipeline + obs.get("incoming_shipment_now", 0)
    order = max(0, order_up_to - inventory_position)
    return int(max(0, min(round(order), FIXED_PARAMS["max_order_amount"])))

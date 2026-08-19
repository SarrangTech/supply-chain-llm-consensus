"""
Frameworks (a) and (c): Standalone LLM agents, with and without tool usage.
Figure 3(a)/(c), Figure 4. No inter-agent communication, so no LangGraph
orchestration is needed here -- the paper only uses LangGraph for the
communication frameworks (Section 4.2).
"""
from ..config import FIXED_PARAMS
from ..prompts import build_agent_prompt
from ..tools import demand_forecast_tool, eoq_tool


def make_standalone_decision_fn(clients: dict, model_tier: str, metric: str, use_tool: bool):
    """
    clients: {agent_idx: LLMClient-like object}
    model_tier: "small" or "large" (controls the P6 tool-emphasis wording)
    metric: "cost" or "bullwhip"
    use_tool: False -> framework (a); True -> framework (c)
    """

    def decision_fn(env) -> list[int]:
        orders = []
        for i in range(env.num_agents):
            obs = env.observe(i)
            agent = env.agents[i]
            total_cost_so_far = sum(agent.cost_history)

            tool_name = None
            tool_output = None
            if use_tool:
                if metric == "cost":
                    tool_name = "Demand Forecasting with Linear Regression"
                    tool_output = demand_forecast_tool(
                        agent.demand_history, lookback=FIXED_PARAMS["demand_forecast_lookback"]
                    )
                else:
                    tool_name = "Economic Order Quantity (EOQ)"
                    tool_output = eoq_tool(
                        agent.demand_history,
                        ordering_cost=FIXED_PARAMS["ordering_cost"],
                        holding_cost=FIXED_PARAMS["inventory_cost"],
                        lookback=FIXED_PARAMS["demand_forecast_lookback"],
                    )

            prompt = build_agent_prompt(
                metric=metric,
                obs=obs,
                total_cost_so_far=total_cost_so_far,
                inventory_hist=obs["inventory_history"],
                backlog_hist=obs["backlog_history"],
                order_hist=obs["own_order_history"],
                demand_hist=obs["downstream_demand_history"],
                lead_time=env.lead_time,
                model_tier=model_tier,
                tool_name=tool_name,
                tool_output=tool_output,
            )
            order = clients[i].get_order_decision(prompt, max_order=env.max_order_amount)
            orders.append(order)
        return orders

    return decision_fn

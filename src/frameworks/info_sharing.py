"""
Frameworks (b) and (d): information sharing between neighbouring agents,
with and without tool usage. Figure 3(b)/(d), Figure 6 (LangGraph):

    start -> Agent-to-agent info sharing -> Agents' final decision
          -> Agent summarises -> end

Per Section 4.2: "each agent initiates communication with its immediate
upstream neighbour... A single interaction, therefore, always involves two
neighbouring agents." For a chain of length N this graph is invoked once per
adjacent pair (0,1), (1,2), ..., (N-2,N-1) each timestep, dynamically, for
any chain length -- mirroring the paper's claim that "our implementation
dynamically creates nodes and edges for a sequential supply chain of any
length" (Section 4.2.1).

FLAGGED JUDGMENT CALL -- reconciling a middle agent's two pairwise decisions:
A middle agent (e.g. agent 1 in a 3-echelon chain) takes part in two pairs
per step: (0,1) as the "upstream" party, and (1,2) as the "downstream"
party. The paper states only that "the final decision stage reconciles
these negotiations" (Section 6.2.1) without giving the reconciliation rule.
We process pairs strictly downstream-to-upstream, (0,1) then (1,2) then...,
and take an agent's LAST-computed pairwise decision within the step as
authoritative (i.e. the decision made using its most upstream-informed
context). This is documented, not inferred from an unseen implementation.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from ..config import FIXED_PARAMS
from ..prompts import build_agent_prompt, neighbor_info_text
from ..tools import demand_forecast_tool, eoq_tool


class PairState(TypedDict, total=False):
    metric: str
    use_tool: bool
    downstream_idx: int
    upstream_idx: int
    downstream_obs: dict
    upstream_obs: dict
    downstream_tool: float | None
    upstream_tool: float | None
    downstream_tool_name: str | None
    downstream_model_tier: str
    upstream_model_tier: str
    downstream_client: object
    upstream_client: object
    downstream_hist: dict
    upstream_hist: dict
    downstream_total_cost: float
    upstream_total_cost: float
    lead_time: int
    max_order: int
    downstream_order: int
    upstream_order: int
    summary: str


def _node_info_sharing(state: PairState) -> PairState:
    # Nothing to compute here beyond what's already in state; this node
    # exists to mirror Figure 6's explicit "Agent-to-agent info sharing"
    # step, where each agent's raw observation (and tool output, if any)
    # becomes visible to its neighbour.
    return state


def _node_final_decision(state: PairState) -> PairState:
    d_neighbor_text = neighbor_info_text("upstream", state["upstream_obs"], state.get("upstream_tool"))
    u_neighbor_text = neighbor_info_text("downstream", state["downstream_obs"], state.get("downstream_tool"))

    d_prompt = build_agent_prompt(
        metric=state["metric"],
        obs=state["downstream_obs"],
        total_cost_so_far=state["downstream_total_cost"],
        inventory_hist=state["downstream_hist"]["inventory"],
        backlog_hist=state["downstream_hist"]["backlog"],
        order_hist=state["downstream_hist"]["order"],
        demand_hist=state["downstream_obs"]["downstream_demand_history"],
        lead_time=state["lead_time"],
        model_tier=state["downstream_model_tier"],
        tool_name=state.get("downstream_tool_name") if state["use_tool"] else None,
        tool_output=state.get("downstream_tool") if state["use_tool"] else None,
        neighbor_info_text=d_neighbor_text,
    )
    u_prompt = build_agent_prompt(
        metric=state["metric"],
        obs=state["upstream_obs"],
        total_cost_so_far=state["upstream_total_cost"],
        inventory_hist=state["upstream_hist"]["inventory"],
        backlog_hist=state["upstream_hist"]["backlog"],
        order_hist=state["upstream_hist"]["order"],
        demand_hist=state["upstream_obs"]["downstream_demand_history"],
        lead_time=state["lead_time"],
        model_tier=state["upstream_model_tier"],
        tool_name=state.get("downstream_tool_name") if state["use_tool"] else None,
        tool_output=state.get("upstream_tool") if state["use_tool"] else None,
        neighbor_info_text=u_neighbor_text,
    )

    state["downstream_order"] = state["downstream_client"].get_order_decision(d_prompt, max_order=state["max_order"])
    state["upstream_order"] = state["upstream_client"].get_order_decision(u_prompt, max_order=state["max_order"])
    return state


def _node_summarise(state: PairState) -> PairState:
    state["summary"] = (
        f"Pair ({state['downstream_idx']},{state['upstream_idx']}): "
        f"downstream ordered {state['downstream_order']}, upstream ordered {state['upstream_order']}."
    )
    return state


def _build_pair_graph():
    g = StateGraph(PairState)
    g.add_node("info_sharing", _node_info_sharing)
    g.add_node("final_decision", _node_final_decision)
    g.add_node("summarise", _node_summarise)
    g.add_edge(START, "info_sharing")
    g.add_edge("info_sharing", "final_decision")
    g.add_edge("final_decision", "summarise")
    g.add_edge("summarise", END)
    return g.compile()


_PAIR_GRAPH = _build_pair_graph()


def make_info_sharing_decision_fn(clients: dict, model_tiers: dict, metric: str, use_tool: bool):
    """
    clients: {agent_idx: LLMClient-like}
    model_tiers: {agent_idx: "small"|"large"}
    """

    def decision_fn(env) -> list[int]:
        final_orders = {}

        for d_idx in range(env.num_agents - 1):
            u_idx = d_idx + 1
            d_obs = env.observe(d_idx)
            u_obs = env.observe(u_idx)
            d_agent, u_agent = env.agents[d_idx], env.agents[u_idx]

            d_tool = u_tool = None
            tool_name = None
            if use_tool:
                if metric == "cost":
                    tool_name = "Demand Forecasting with Linear Regression"
                    d_tool = demand_forecast_tool(d_agent.demand_history, lookback=FIXED_PARAMS["demand_forecast_lookback"])
                    u_tool = demand_forecast_tool(u_agent.demand_history, lookback=FIXED_PARAMS["demand_forecast_lookback"])
                else:
                    tool_name = "Economic Order Quantity (EOQ)"
                    d_tool = eoq_tool(d_agent.demand_history, FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], FIXED_PARAMS["demand_forecast_lookback"])
                    u_tool = eoq_tool(u_agent.demand_history, FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], FIXED_PARAMS["demand_forecast_lookback"])

            init_state: PairState = {
                "metric": metric,
                "use_tool": use_tool,
                "downstream_idx": d_idx,
                "upstream_idx": u_idx,
                "downstream_obs": d_obs,
                "upstream_obs": u_obs,
                "downstream_tool": d_tool,
                "upstream_tool": u_tool,
                "downstream_tool_name": tool_name,
                "downstream_model_tier": model_tiers[d_idx],
                "upstream_model_tier": model_tiers[u_idx],
                "downstream_client": clients[d_idx],
                "upstream_client": clients[u_idx],
                "downstream_hist": {
                    "inventory": d_obs["inventory_history"],
                    "backlog": d_obs["backlog_history"],
                    "order": d_obs["own_order_history"],
                },
                "upstream_hist": {
                    "inventory": u_obs["inventory_history"],
                    "backlog": u_obs["backlog_history"],
                    "order": u_obs["own_order_history"],
                },
                "downstream_total_cost": sum(d_agent.cost_history),
                "upstream_total_cost": sum(u_agent.cost_history),
                "lead_time": env.lead_time,
                "max_order": env.max_order_amount,
            }

            result = _PAIR_GRAPH.invoke(init_state)
            final_orders[d_idx] = result["downstream_order"]
            final_orders[u_idx] = result["upstream_order"]  # overwritten by the next pair if this agent is a "downstream" party next

        return [final_orders[i] for i in range(env.num_agents)]

    return decision_fn

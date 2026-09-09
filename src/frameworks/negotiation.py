"""
Framework (e): negotiation between neighbouring agents around tool output.
Figure 3(e), Figure 7 (LangGraph):

    start -> Agent-to-agent info sharing
          -> {Agent initiates <-> Agent responds} x num_iter
          -> Agents' final decision -> Agent summarises -> end

num_iter = 3 (Appendix 6, Table 2): "each agent pair has 3 back-and-forth
passes before moving to the agreement stage" (Table 2 footnote).

Per Section 3.2 / 6.2.1: the tool output (EOQ for bullwhip experiments,
demand-forecast prediction for cost experiments -- Section 4.3.2 says both
metrics use the same framework machinery) serves as the upper/lower bound
that anchors the negotiation range. The downstream agent starts the
conversation (Section 6.2.1: "the downstream agent initiates the
conversation").

FLAGGED JUDGMENT CALL: Figure 9's transcript shows the conversation intro
("Let's have a conversation about how much to order...") as free-form LLM
output, not a fixed template. We reproduce this by having the downstream
agent's own LLM generate that opening line (seeded with a fixed instruction
prompt, `negotiation_intro_prompt`), rather than hard-coding the sentence
verbatim -- consistent with "our frameworks... govern how LLM agents
perceive, communicate, and act" (Section 1) rather than scripting their
words for them.

FLAGGED JUDGMENT CALL (EOQ per bilateral pair): as documented in
info_sharing.py and in NOTES_AND_ASSUMPTIONS.md, each agent's tool output is
computed once per timestep from its own downstream_demand_history, and reused
for whichever negotiation pair it participates in during that timestep. The
paper's text describing "different EOQ per pair" (Section 6.2.1) is not
followed literally due to ambiguity in how exactly it differs; see notes.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from ..config import FIXED_PARAMS
from ..prompts import negotiation_intro_prompt, negotiation_turn_prompt, negotiation_final_question_prompt
from ..tools import demand_forecast_tool, eoq_tool

NUM_ITER = FIXED_PARAMS["num_negotiation_iters"]


class NegotiationState(TypedDict, total=False):
    downstream_idx: int
    upstream_idx: int
    downstream_client: object
    upstream_client: object
    downstream_eoq: float
    upstream_eoq: float
    transcript: list  # list[tuple[str, str]] of (speaker, text)
    counter: int
    downstream_order: int
    upstream_order: int
    downstream_final_reply: str
    upstream_final_reply: str
    max_order: int


def _node_info_sharing(state: NegotiationState) -> NegotiationState:
    lower = min(state["downstream_eoq"], state["upstream_eoq"])
    upper = max(state["downstream_eoq"], state["upstream_eoq"])
    intro_prompt = negotiation_intro_prompt(state["downstream_eoq"], lower, upper)
    opening = state["downstream_client"].chat(intro_prompt)
    state["transcript"] = [("downstream", opening)]
    state["counter"] = 0
    return state


def _node_agent_initiates(state: NegotiationState) -> NegotiationState:
    # After the first loop, the downstream agent speaks again in response to
    # the upstream agent's most recent message.
    last_msg = state["transcript"][-1][1]
    prompt = negotiation_turn_prompt(last_msg, state["downstream_eoq"])
    reply = state["downstream_client"].chat(prompt)
    state["transcript"].append(("downstream", reply))
    return state


def _node_agent_responds(state: NegotiationState) -> NegotiationState:
    last_msg = state["transcript"][-1][1]
    prompt = negotiation_turn_prompt(last_msg, state["upstream_eoq"])
    reply = state["upstream_client"].chat(prompt)
    state["transcript"].append(("upstream", reply))
    state["counter"] += 1
    return state


def _loop_or_finalise(state: NegotiationState) -> str:
    return "agent_initiates" if state["counter"] < NUM_ITER else "final_decision"


def _node_final_decision(state: NegotiationState) -> NegotiationState:
    from ..llm_client import parse_order_answer

    def _extract_or_ask_again(client, own_eoq) -> tuple[int, str]:
        # Ask explicitly for the final numeric answer, per Figure 9's
        # "System: What is your final answer?" step. strict_format=True
        # here (unlike the earlier free-form negotiation turns) since this
        # reply must itself contain a parseable [[N]] answer.
        #
        # Grounded in the actual transcript (see prompts.py's
        # negotiation_final_question_prompt docstring / NOTES_AND_ASSUMPTIONS.md
        # section (g)) -- BaseChatClient.chat() has no memory between calls,
        # so without this the model answered blind, disconnected from
        # whatever it just negotiated.
        final_question = negotiation_final_question_prompt(state["transcript"], own_eoq)
        reply = client.chat(final_question, strict_format=True)
        # parse_order_answer (section (h)) tolerates reasoning shown inside
        # the brackets -- grounding the question in the transcript measurably
        # made the model more likely to imitate the conversation's own
        # shown-work style even here, breaking the bare-[[N]] assumption.
        value = parse_order_answer(reply, state["max_order"])
        if value is not None:
            return value, reply
        # Fall back to the structured single-shot decision call if even the
        # lenient extraction found no number at all.
        import sys
        model = getattr(client, "model", "?")
        print(f"NEGOTIATION_FINAL_FALLBACK model={model} unparseable_reply={reply!r}", file=sys.stderr, flush=True)
        value = client.get_order_decision(final_question, max_order=state["max_order"])
        return value, reply

    state["downstream_order"], state["downstream_final_reply"] = _extract_or_ask_again(state["downstream_client"], state["downstream_eoq"])
    state["upstream_order"], state["upstream_final_reply"] = _extract_or_ask_again(state["upstream_client"], state["upstream_eoq"])
    return state


def _node_summarise(state: NegotiationState) -> NegotiationState:
    return state


def _build_negotiation_graph():
    g = StateGraph(NegotiationState)
    g.add_node("info_sharing", _node_info_sharing)
    g.add_node("agent_initiates", _node_agent_initiates)
    g.add_node("agent_responds", _node_agent_responds)
    g.add_node("final_decision", _node_final_decision)
    g.add_node("summarise", _node_summarise)

    g.add_edge(START, "info_sharing")
    g.add_edge("info_sharing", "agent_responds")  # first upstream reply to the opening
    g.add_conditional_edges("agent_responds", _loop_or_finalise, {
        "agent_initiates": "agent_initiates",
        "final_decision": "final_decision",
    })
    g.add_edge("agent_initiates", "agent_responds")
    g.add_edge("final_decision", "summarise")
    g.add_edge("summarise", END)
    return g.compile()


_NEGOTIATION_GRAPH = _build_negotiation_graph()


def make_negotiation_decision_fn(clients: dict, model_tiers: dict, metric: str, transcript_sink: list | None = None):
    """
    transcript_sink: if given, every pairwise negotiation's full turn-by-turn
    transcript (plus the step index and both final orders) is appended to
    it. Diagnostic use only (see NOTES_AND_ASSUMPTIONS.md section (g)) --
    the real conversation text is otherwise computed and immediately
    discarded, which made it impossible to tell *why* an aggregate cost/
    bullwhip number looked off without rerunning with this on.
    """
    def decision_fn(env) -> list[int]:
        final_orders = {}

        for d_idx in range(env.num_agents - 1):
            u_idx = d_idx + 1
            d_agent, u_agent = env.agents[d_idx], env.agents[u_idx]

            if metric == "cost":
                d_tool = demand_forecast_tool(d_agent.demand_history, lookback=FIXED_PARAMS["demand_forecast_lookback"])
                u_tool = demand_forecast_tool(u_agent.demand_history, lookback=FIXED_PARAMS["demand_forecast_lookback"])
            else:
                d_tool = eoq_tool(d_agent.demand_history, FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], FIXED_PARAMS["demand_forecast_lookback"])
                u_tool = eoq_tool(u_agent.demand_history, FIXED_PARAMS["ordering_cost"], FIXED_PARAMS["inventory_cost"], FIXED_PARAMS["demand_forecast_lookback"])

            init_state: NegotiationState = {
                "downstream_idx": d_idx,
                "upstream_idx": u_idx,
                "downstream_client": clients[d_idx],
                "upstream_client": clients[u_idx],
                "downstream_eoq": d_tool,
                "upstream_eoq": u_tool,
                "max_order": env.max_order_amount,
            }
            result = _NEGOTIATION_GRAPH.invoke(init_state)
            final_orders[d_idx] = result["downstream_order"]
            final_orders[u_idx] = result["upstream_order"]

            if transcript_sink is not None:
                transcript_sink.append({
                    "step": env.t,
                    "pair": [d_idx, u_idx],
                    "downstream_eoq": d_tool,
                    "upstream_eoq": u_tool,
                    "transcript": result["transcript"],
                    "downstream_order": result["downstream_order"],
                    "upstream_order": result["upstream_order"],
                    "downstream_final_reply": result["downstream_final_reply"],
                    "upstream_final_reply": result["upstream_final_reply"],
                })

        return [final_orders[i] for i in range(env.num_agents)]

    return decision_fn

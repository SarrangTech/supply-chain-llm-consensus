"""
Prompt construction, following the component breakdown in Appendix 2
(Figure 13) and the Flash-vs-Pro divergence in Appendix 3 (Figure 14).

Components, in the order the paper lists them (Section 4.1.1):
  P1: general description of the sequential supply chain problem setting
  P2: description of the objective function (cost, or coefficient of variation)
  P3: the agent's own observation of the environment
  P4: memory -- previous 10 observations of inventory/backlog/order amounts
  P5: additional info (recent demand history; neighbour-shared info, when applicable)
  P6: tool output, with contextualisation (only when a tool is used)
  P7: final question on order amount + output-formatting instruction

FLAGGED JUDGMENT CALL: the paper shows one illustrative filled-in prompt
(Figure 13) for the cost-minimisation, standalone-with-tool case. The exact
wording for the bullwhip metric, for information-sharing, and for
negotiation is not shown verbatim -- only described narratively (Section 4,
Figure 9's negotiation transcript gives some exact phrasing, which we reuse
directly where given). Everything else below is written to match the
paper's described structure and content as closely as possible, not quoted
verbatim from an unseen original prompt.
"""
from .config import FIXED_PARAMS

MAX_ORDER = FIXED_PARAMS["max_order_amount"]

# ---------------------------------------------------------------------------
# P1 + P2 -- problem setting and objective function, per metric.
# ---------------------------------------------------------------------------

_P1 = (
    "This is a simulation of a sequential supply chain. Your overall goal is to "
    "minimize the {objective_noun} for the whole supply chain, not just your local "
    "costs. So don't act too selfishly. At the same time, you must balance inventory "
    "levels to meet demand while minimizing holding costs. Holding excess inventory is "
    "very costly, and there are penalties for backlog, simulating lost sales or unhappy "
    "customers. Random demand fluctuations are included to simulate market conditions, "
    "requiring adaptive inventory strategies. High holding costs for excess inventory "
    "can lead to significant financial losses."
)

_P2_COST = (
    "Consider the objective function to achieve: Minimize the total cost, where the "
    "total cost is the sum of:\n"
    "+ Holding cost: inventory level * (1)\n"
    "+ Backlog cost: backlog level * (1)\n"
    "+ Ordering cost: order quantity * (1)\n"
    "+ Fixed cost: a fixed cost (= 1) iff order quantity > 0."
)

_P2_BULLWHIP = (
    "Consider the objective function to achieve: Minimize the bullwhip effect, "
    "measured as the coefficient of variation of your own order quantities over time:\n"
    "BW = std(your_orders) / mean(your_orders).\n"
    "A coefficient of variation below 1 means the bullwhip effect is negligible. Avoid "
    "large swings in your order amounts relative to their average."
)


def _p1_p2(metric: str) -> str:
    objective_noun = "total cost" if metric == "cost" else "bullwhip effect"
    p1 = _P1.format(objective_noun=objective_noun)
    p2 = _P2_COST if metric == "cost" else _P2_BULLWHIP
    return p1 + "\n\n" + p2


# ---------------------------------------------------------------------------
# P3 -- agent's own observation.
# ---------------------------------------------------------------------------

def _p3(obs: dict, total_cost_so_far: float) -> str:
    return (
        "To make this decision, here are the current observations for this agent:\n"
        f"Inventory level: {obs['inventory']}\n"
        f"Backlog: {obs['backlog']}\n"
        f"Latest order: {obs['last_order']}\n"
        "Inventory level is the amount the agent has in stock, whereas the orders "
        "that are on their way to the agent in the next steps are: "
        f"{obs['upcoming_shipments']}.\n"
        f"Total cost for this agent so far: {total_cost_so_far}"
    )


# ---------------------------------------------------------------------------
# P4 -- memory (previous 10 steps).
# ---------------------------------------------------------------------------

def _p4(inventory_hist: list, backlog_hist: list, order_hist: list) -> str:
    n = min(len(inventory_hist), len(backlog_hist), len(order_hist))
    lines = []
    for i in range(n):
        lines.append(
            f'{{"current_inventory":{inventory_hist[i]},"current_backlog":{backlog_hist[i]},'
            f'"latest_order":{order_hist[i]}}}'
        )
    return "Here are the variable observations for the last " + str(n) + " time steps:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# P5 -- recent downstream demand history, and (optionally) shared neighbour info.
# ---------------------------------------------------------------------------

def _p5(demand_hist: list, lead_time: int) -> str:
    window = demand_hist[-10:]
    return (
        f"Last {len(window)} steps showed a demand of {window} from your downstream neighbor. "
        f"The action (amount ordered) will arrive in {lead_time} time steps."
    )


def neighbor_info_text(neighbor_role: str, neighbor_obs: dict, neighbor_tool_output: float | None) -> str:
    parts = [
        f"Information shared by your {neighbor_role} neighbor:",
        f"  Their inventory level: {neighbor_obs['inventory']}",
        f"  Their backlog: {neighbor_obs['backlog']}",
        f"  Their latest order: {neighbor_obs['last_order']}",
    ]
    if neighbor_tool_output is not None:
        parts.append(f"  Their tool output: {neighbor_tool_output:.2f}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# P6 -- tool output, with contextualisation. Larger models get the extra
# "use as an upper bound" instruction per Appendix 3 / Figure 14.
# ---------------------------------------------------------------------------

def _p6(tool_name: str, tool_output: float, model_tier: str) -> str:
    base = (
        f"The [{tool_name}] tool has been activated. The result obtained from this "
        f"tool is: [{tool_output:.2f}]. Please give a lot of weight to this "
        "forecast in your decision-making process."
    )
    if model_tier == "large":
        # Appendix 3 (Figure 14): "Gemini Pro required more insistence in the
        # prompt to steer the decision-making closer to the tool output."
        # We apply the analogous stronger instruction to our "large" tier
        # model (Llama 3.1 70B, standing in for Gemini 1.5 Pro).
        base += " Use it as an upper bound for the final output."
    return base


# ---------------------------------------------------------------------------
# P7 -- final question and output formatting.
# ---------------------------------------------------------------------------

_P7 = (
    "What should be the next amount to order? "
    "Please provide only a number in the form of an integer value where integer value "
    f"is a number from 0 to {MAX_ORDER}. Enclose your numerical answer within triple "
    "brackets, for example: [[2]]. Do not show your reasoning or explanation -- your "
    "entire reply must be only the bracketed number, nothing else."
)


def build_agent_prompt(
    metric: str,
    obs: dict,
    total_cost_so_far: float,
    inventory_hist: list,
    backlog_hist: list,
    order_hist: list,
    demand_hist: list,
    lead_time: int,
    model_tier: str,
    tool_name: str | None = None,
    tool_output: float | None = None,
    neighbor_info_text: str | None = None,
) -> str:
    """
    Assemble the full P1-P7 prompt for one agent's decision at one timestep.
    `tool_name`/`tool_output` are set only for the *_tool frameworks.
    `neighbor_info_text` is set only for the info_sharing / negotiation
    frameworks (built via _p5_neighbor_info before calling this).
    """
    sections = [
        _p1_p2(metric),
        _p3(obs, total_cost_so_far),
        _p4(inventory_hist, backlog_hist, order_hist),
        _p5(demand_hist, lead_time),
    ]
    if neighbor_info_text is not None:
        sections.append(neighbor_info_text)
    if tool_name is not None and tool_output is not None:
        sections.append(_p6(tool_name, tool_output, model_tier))
    sections.append(_P7)
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Negotiation-specific prompts (Section 6.2.1 / Figures 9-10 give exact
# phrasing for the conversation intro and turn structure; reused verbatim
# where the paper shows it).
# ---------------------------------------------------------------------------

def negotiation_intro_prompt(own_eoq: float, lower_bound: float, upper_bound: float) -> str:
    # Verbatim structure per Figure 9's transcript.
    return (
        "Let's have a conversation about how much to order to minimize the overall "
        "bullwhip effect. Let's use the EOQs shared by each agent as upper and lower "
        f"bound for the negotiation. My EOQ is {own_eoq:.2f}; I am willing to negotiate "
        "how much to order in order to minimize the bullwhip effect."
    )


def negotiation_turn_prompt(counterpart_message: str, own_eoq: float) -> str:
    return (
        f'Your neighbor said: "{counterpart_message}"\n'
        f"Your own EOQ is {own_eoq:.2f}. Respond with your proposal or agreement, "
        "working toward a compromise order amount that minimizes the bullwhip effect."
    )


NEGOTIATION_FINAL_QUESTION = (
    "What is your final answer? Please provide only a number in the form of an "
    f"integer value from 0 to {MAX_ORDER}. Enclose your numerical answer within "
    "triple brackets, for example: [[2]]. Do not show your reasoning or explanation "
    "-- your entire reply must be only the bracketed number, nothing else."
)


def negotiation_final_question_prompt(transcript: list[tuple[str, str]], own_eoq: float) -> str:
    """
    BUG FIX (see NOTES_AND_ASSUMPTIONS.md section (g)): BaseChatClient.chat()
    is a stateless, single-turn call by design (no conversation history is
    kept between calls) -- every earlier version of this final question was
    sent with NO memory of the negotiation that just happened, only the
    agent's own EOQ. Measured effect: 62/400 (15.5%) of negotiation sessions
    in a real 200-step run produced a final answer clamped to the hard
    max-order cap (100), completely disconnected from an otherwise coherent,
    converging conversation -- because the model had no way to know what was
    actually discussed or agreed. This grounds the final question in the
    transcript explicitly, since nothing else carries it forward.
    """
    convo = "\n".join(f'{speaker}: "{text}"' for speaker, text in transcript)
    return (
        "Here is the negotiation conversation you just had:\n\n"
        f"{convo}\n\n"
        + NEGOTIATION_FINAL_QUESTION
        + f" (Your own EOQ was {own_eoq:.2f}.)"
    )

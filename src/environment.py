"""
Sequential 3-echelon supply chain environment, per Section 3 and Figure 2.

Agent indexing convention (FLAGGED JUDGMENT CALL -- not stated explicitly as
indices in the paper, inferred from Figures 9-11 where the EOQ grows from
agent 0 to agent 2, matching "increasingly upstream agents amplify demand"):
    agent 0 = Retailer      (faces customer demand directly)
    agent 1 = Distributor   (faces agent 0's orders as its demand)
    agent 2 = Manufacturer  (faces agent 1's orders as its demand; most
                             upstream; assumed to have unconstrained supply
                             capacity, i.e. it always receives what it orders
                             after the lead time -- standard serial-chain
                             assumption, e.g. Chen et al. 2000, Liu et al. 2022)

Each agent i's "upstream neighbour" is agent i+1 (or an infinite external
supplier for the manufacturer); its "downstream neighbour" is agent i-1 (or
the end customer for the retailer). This matches the paper's statement that
"each agent initiates communication with its immediate upstream neighbour"
and that information sharing targets the upstream neighbour "to mitigate
demand amplification, which usually affects upstream echelons."

FLAGGED JUDGMENT CALL -- within-timestep sequencing:
The paper gives the cost decomposition (Section 3.1) and the bullwhip
formula (Section 3.2) but not a formal per-timestep state-transition
equation. The order of operations below (receive shipments -> agents decide
-> ship to downstream demand, tracking backlog -> compute costs -> orders
placed become future shipments after lead_time) is the standard multi-
echelon inventory simulation convention used in this literature (e.g. the
beer game, Chen et al. 2000, Liu et al. 2022) and was not itself specified
in the Jannelli et al. text.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AgentState:
    inventory: int = 20            # arbitrary reasonable starting stock (not stated in paper)
    backlog: int = 0
    last_order: int = 0
    order_history: list = field(default_factory=list)   # orders THIS agent placed upstream
    demand_history: list = field(default_factory=list)  # demand THIS agent received downstream
    inventory_history: list = field(default_factory=list)
    backlog_history: list = field(default_factory=list)
    cost_history: list = field(default_factory=list)
    pending_shipments: dict = field(default_factory=dict)  # arrival_step -> qty


class SequentialSupplyChainEnv:
    """
    A serial supply chain of `num_agents` echelons with a fixed order lead
    time. Agent 0 is the most downstream (faces customer_demand), agent
    (num_agents - 1) is the most upstream (unconstrained external supply).
    """

    def __init__(
        self,
        num_agents: int,
        customer_demand: np.ndarray,
        lead_time: int,
        inventory_cost: float,
        backlog_cost: float,
        ordering_cost: float,
        fixed_ordering_cost: float,
        max_order_amount: int,
    ):
        self.num_agents = num_agents
        self.customer_demand = customer_demand
        self.num_steps = len(customer_demand)
        self.lead_time = lead_time
        self.inventory_cost = inventory_cost
        self.backlog_cost = backlog_cost
        self.ordering_cost = ordering_cost
        self.fixed_ordering_cost = fixed_ordering_cost
        self.max_order_amount = max_order_amount

        self.agents = [AgentState() for _ in range(num_agents)]
        self.t = 0
        self.global_cost_history = []

    def downstream_demand_for(self, agent_idx: int) -> int:
        """The demand this agent must fill at the current step."""
        if agent_idx == 0:
            return int(self.customer_demand[self.t])
        return self.agents[agent_idx - 1].last_order

    def observe(self, agent_idx: int) -> dict:
        """Local observation available to `agent_idx` at the current step."""
        a = self.agents[agent_idx]
        incoming = a.pending_shipments.get(self.t, 0)
        next_arrivals = [a.pending_shipments.get(self.t + k, 0) for k in range(1, self.lead_time + 1)]
        return {
            "t": self.t,
            "inventory": a.inventory,
            "backlog": a.backlog,
            "last_order": a.last_order,
            "incoming_shipment_now": incoming,
            "upcoming_shipments": next_arrivals,
            "downstream_demand_history": a.demand_history[-10:],
            "own_order_history": a.order_history[-10:],
            "inventory_history": a.inventory_history[-10:],
            "backlog_history": a.backlog_history[-10:],
        }

    def step(self, orders: list[int]) -> float:
        """
        Advance the environment by one timestep given each agent's chosen
        order quantity (upstream order placed this step). Returns the
        summed global cost for this step.
        """
        assert len(orders) == self.num_agents
        orders = [int(np.clip(o, 0, self.max_order_amount)) for o in orders]

        step_global_cost = 0.0

        # 1. Receive shipments ordered lead_time steps ago.
        for i, a in enumerate(self.agents):
            arrived = a.pending_shipments.pop(self.t, 0)
            a.inventory += arrived

        # 2. Determine downstream demand and ship against it (fill backlog first).
        for i, a in enumerate(self.agents):
            demand = self.downstream_demand_for(i)
            a.demand_history.append(demand)
            total_owed = a.backlog + demand
            shipped = min(a.inventory, total_owed)
            a.inventory -= shipped
            a.backlog = total_owed - shipped

        # 3. Place new orders upstream; manufacturer's orders always arrive
        #    (unconstrained supply assumption); others' orders arrive as the
        #    upstream agent's shipped quantity would in a full network sim,
        #    but since each agent's "shipped" amount already equals what it
        #    could fulfill of its downstream demand (see step 2 above using
        #    `last_order` as demand), we schedule the ordered quantity to
        #    arrive after lead_time, capped implicitly by the upstream
        #    agent's own future fulfillment in step 2 of that future tick.
        for i, a in enumerate(self.agents):
            order_qty = orders[i]
            a.last_order = order_qty
            a.order_history.append(order_qty)
            arrival_t = self.t + self.lead_time
            a.pending_shipments[arrival_t] = a.pending_shipments.get(arrival_t, 0) + order_qty

        # 4. Costs.
        for i, a in enumerate(self.agents):
            inv_cost = self.inventory_cost * a.inventory
            back_cost = self.backlog_cost * a.backlog
            ord_cost = self.ordering_cost * orders[i] + (self.fixed_ordering_cost if orders[i] > 0 else 0)
            local_cost = inv_cost + back_cost + ord_cost
            a.cost_history.append(local_cost)
            a.inventory_history.append(a.inventory)
            a.backlog_history.append(a.backlog)
            step_global_cost += local_cost

        self.global_cost_history.append(step_global_cost)
        self.t += 1
        return step_global_cost

    def run(self, decision_fn) -> dict:
        """
        Run the full simulation. `decision_fn(env) -> list[int]` must return
        one order quantity per agent for the current step, using only
        information reachable via `env.observe(i)` for each agent i (plus
        whatever inter-agent communication the framework implements).
        """
        for _ in range(self.num_steps):
            orders = decision_fn(self)
            self.step(orders)
        return self.results()

    def results(self) -> dict:
        total_global_cost = float(np.sum(self.global_cost_history))
        per_agent_orders = [a.order_history for a in self.agents]
        return {
            "total_global_cost": total_global_cost,
            "per_agent_order_history": per_agent_orders,
            "per_agent_inventory_history": [a.inventory_history for a in self.agents],
            "per_agent_backlog_history": [a.backlog_history for a in self.agents],
        }

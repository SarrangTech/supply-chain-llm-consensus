"""
Metrics, Section 3.1 (global cost) and Section 3.2 (bullwhip effect).

Bullwhip per echelon: BW_i = sigma(d_i) / mu(d_i), the coefficient of
variation of agent i's OWN order history (its "demand" placed on its
upstream neighbour), computed over the full run.

Aggregate bullwhip: the paper states "the aggregate bullwhip effect for a
(strictly) end-to-end supply chain can be achieved by multiplying all of the
coefficients of variation ... of each individual echelon" (Fransoo and
Wouters 2000), i.e. BW_global = product_i(BW_i).

A coefficient of variation below 1 indicates the bullwhip effect is
negligible for that echelon (paper, Section 3.2).
"""
import numpy as np


def coefficient_of_variation(series: list[float]) -> float:
    arr = np.array(series, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std() / mean)


def per_agent_bullwhip(per_agent_order_history: list[list[int]]) -> list[float]:
    return [coefficient_of_variation(orders) for orders in per_agent_order_history]


def global_bullwhip(per_agent_order_history: list[list[int]]) -> float:
    bw = per_agent_bullwhip(per_agent_order_history)
    product = 1.0
    for b in bw:
        product *= b
    return product


def global_cost(env_results: dict) -> float:
    return env_results["total_global_cost"]

"""
The two agent tools from Section 4.4:

1. Demand forecasting tool (cost-minimisation experiments): linear regression
   over the last 30 observed periods, predicting the next order amount.
   "At the beginning of the simulation, when the dataset is insufficient, the
   agent defaults to the most recently observed order quantity."

2. EOQ tool (bullwhip-effect experiments): represents each agent's selfish
   incentive.
       EOQ = sqrt(2 * Demand * OrderingCost / HoldingCost)
   where Demand is the mean of historical demand from the downstream agent.
"""
import numpy as np


def demand_forecast_tool(demand_history: list[int], lookback: int = 30) -> float:
    """
    Predict the next order amount via linear regression over the last
    `lookback` observed demand periods. Falls back to the most recent
    observation if fewer than `lookback` observations are available yet.
    """
    if len(demand_history) == 0:
        return 0.0
    if len(demand_history) < lookback:
        return float(demand_history[-1])

    window = np.array(demand_history[-lookback:], dtype=float)
    x = np.arange(len(window))
    # Ordinary least squares fit: window ~ a * x + b
    a, b = np.polyfit(x, window, deg=1)
    next_x = len(window)  # predict one step beyond the window
    prediction = a * next_x + b
    return max(0.0, float(prediction))


def eoq_tool(
    demand_history: list[int],
    ordering_cost: float,
    holding_cost: float,
    lookback: int = 30,
) -> float:
    """
    Economic Order Quantity, using the mean of historical demand from the
    downstream agent over the last `lookback` observations (Section 3.2).
    """
    if len(demand_history) == 0:
        return 0.0
    window = demand_history[-lookback:]
    mean_demand = float(np.mean(window))
    if mean_demand <= 0 or holding_cost <= 0:
        return 0.0
    return float(np.sqrt(2 * mean_demand * ordering_cost / holding_cost))

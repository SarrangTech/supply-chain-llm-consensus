"""
Builds and executes the full experimental grid (Table 1 main text +
Appendix 4 Table 1, and Appendix Table 2 for the bullwhip metric), and
formats results in the paper's table layout for direct comparison.

Grid, per the task spec:
  Metric = Global Cost: weak baseline, strong baseline, then
           5 frameworks x 2 model tiers = 10 runs. (12 configs total)
  Metric = Global Bullwhip: weak baseline, EOQ-tool baseline,
           Chen et al. (2000) baseline, then
           5 frameworks x 2 model tiers = 10 runs. (13 configs total)
"""
import json
import time
from dataclasses import asdict

from .baselines import (
    chen_et_al_2000_decision,
    demand_forecast_tool_only_decision,
    eoq_tool_only_decision,
    ss_policy_decision,
)
from .config import BACKENDS, FIXED_PARAMS, MODELS, OLLAMA_MODELS
from .demand import MJDParams, generate_mjd_demand
from .environment import SequentialSupplyChainEnv
from .frameworks.info_sharing import make_info_sharing_decision_fn
from .frameworks.negotiation import make_negotiation_decision_fn
from .frameworks.standalone import make_standalone_decision_fn
from .metrics import global_bullwhip, global_cost


def _new_env(demand=None):
    demand = demand if demand is not None else generate_mjd_demand(FIXED_PARAMS["num_steps"], MJDParams())
    return SequentialSupplyChainEnv(
        num_agents=FIXED_PARAMS["num_agents"],
        customer_demand=demand,
        lead_time=FIXED_PARAMS["lead_time"],
        inventory_cost=FIXED_PARAMS["inventory_cost"],
        backlog_cost=FIXED_PARAMS["backlog_cost"],
        ordering_cost=FIXED_PARAMS["ordering_cost"],
        fixed_ordering_cost=FIXED_PARAMS["fixed_ordering_cost"],
        max_order_amount=FIXED_PARAMS["max_order_amount"],
    )


def _ss_policy_decision_fn(env) -> list[int]:
    return [ss_policy_decision(env.observe(i)) for i in range(env.num_agents)]


def _tool_only_decision_fn(env, metric: str) -> list[int]:
    fn = demand_forecast_tool_only_decision if metric == "cost" else eoq_tool_only_decision
    orders = []
    for i in range(env.num_agents):
        obs = env.observe(i)
        obs["downstream_demand_history"] = env.agents[i].demand_history  # full history for the tool
        orders.append(fn(obs))
    return orders


def _chen_2000_decision_fn(env) -> list[int]:
    all_obs = [env.observe(i) for i in range(env.num_agents)]
    retailer_demand_history = env.agents[0].demand_history  # centralised: retailer's demand shared with all
    return [
        chen_et_al_2000_decision(i, all_obs, retailer_demand_history)
        for i in range(env.num_agents)
    ]


def make_client_factory(use_mock: bool):
    """
    Routes each model tier to whichever backend config.BACKENDS says it
    should use right now (see the comment there for why "small" runs
    locally via Ollama while "large" stays on OpenRouter).
    """
    if use_mock:
        from .mock_llm import MockLLMClient

        def factory(model_tier: str):
            return MockLLMClient(model=MODELS[model_tier], bias=1.0 if model_tier == "small" else 1.05)

        return factory

    def factory(model_tier: str):
        backend = BACKENDS[model_tier]
        if backend == "ollama":
            from .llm_client import OllamaClient

            return OllamaClient(model=OLLAMA_MODELS[model_tier])
        elif backend == "openrouter":
            from .llm_client import LLMClient

            return LLMClient(model=MODELS[model_tier])
        else:
            raise ValueError(f"Unknown backend for tier {model_tier!r}: {backend!r}")

    return factory


def run_single_experiment(config: dict, use_mock: bool = False, demand=None) -> dict:
    """
    config keys: metric ("cost"|"bullwhip"), baseline (name or None),
    framework (name or None), model_tier ("small"|"large"|None for baselines
    that use both agents' tiers identically -- baselines are model-free).
    """
    env = _new_env(demand)
    metric = config["metric"]

    if config.get("baseline") == "ss_policy":
        decision_fn = _ss_policy_decision_fn
    elif config.get("baseline") == "tool_only":
        decision_fn = lambda e: _tool_only_decision_fn(e, metric)
    elif config.get("baseline") == "chen_2000":
        decision_fn = _chen_2000_decision_fn
    else:
        framework = config["framework"]
        model_tier = config["model_tier"]
        client_factory = make_client_factory(use_mock)
        clients = {i: client_factory(model_tier) for i in range(env.num_agents)}
        model_tiers = {i: model_tier for i in range(env.num_agents)}

        if framework == "standalone":
            decision_fn = make_standalone_decision_fn(clients, model_tier, metric, use_tool=False)
        elif framework == "standalone_tool":
            decision_fn = make_standalone_decision_fn(clients, model_tier, metric, use_tool=True)
        elif framework == "info_sharing":
            decision_fn = make_info_sharing_decision_fn(clients, model_tiers, metric, use_tool=False)
        elif framework == "info_sharing_tool":
            decision_fn = make_info_sharing_decision_fn(clients, model_tiers, metric, use_tool=True)
        elif framework == "negotiation_tool":
            decision_fn = make_negotiation_decision_fn(clients, model_tiers, metric)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    t0 = time.time()
    env.run(decision_fn)
    elapsed = time.time() - t0

    results = env.results()
    return {
        "config": config,
        "cost": global_cost(results),
        "bullwhip": global_bullwhip(results["per_agent_order_history"]),
        "elapsed_sec": elapsed,
    }


def build_cost_grid() -> list[dict]:
    grid = [
        {"metric": "cost", "baseline": "ss_policy", "label": "Restocking Policy (S,s)=(100,60)"},
        {"metric": "cost", "baseline": "tool_only", "label": "Demand Forecasting Tool"},
    ]
    for model_tier in ("small", "large"):
        for framework in ("standalone", "info_sharing", "standalone_tool", "info_sharing_tool", "negotiation_tool"):
            grid.append({
                "metric": "cost",
                "framework": framework,
                "model_tier": model_tier,
                "label": f"{framework} / {model_tier}",
            })
    return grid


def build_bullwhip_grid() -> list[dict]:
    grid = [
        {"metric": "bullwhip", "baseline": "ss_policy", "label": "Restocking Policy (S,s)=(100,60)"},
        {"metric": "bullwhip", "baseline": "tool_only", "label": "EOQ tool"},
        {"metric": "bullwhip", "baseline": "chen_2000", "label": "Chen et al. (2000)"},
    ]
    for model_tier in ("small", "large"):
        for framework in ("standalone", "info_sharing", "standalone_tool", "info_sharing_tool", "negotiation_tool"):
            grid.append({
                "metric": "bullwhip",
                "framework": framework,
                "model_tier": model_tier,
                "label": f"{framework} / {model_tier}",
            })
    return grid


def build_full_grid() -> list[dict]:
    """Stable, fixed ordering: index into this list is a config's identity for sharding."""
    return build_cost_grid() + build_bullwhip_grid()


def filter_grid(
    grid: list[dict],
    only_metric: list[str] | None = None,
    only_framework: list[str] | None = None,
    only_model_tier: list[str] | None = None,
    skip_baselines: bool = False,
) -> list[dict]:
    """
    Select a subset of the full grid, so independent shards of it can be run
    as separate OS processes in parallel (configs don't share any state --
    each builds its own env / clients -- so this is safe). The SAME demand
    series is used regardless of which shard runs it, because
    generate_mjd_demand() is deterministic given num_steps (fixed seed), so
    results stay apples-to-apples across shards without needing to share
    any runtime state between processes.
    """
    out = grid
    if only_metric:
        out = [c for c in out if c["metric"] in only_metric]
    if skip_baselines:
        out = [c for c in out if "baseline" not in c]
    if only_framework:
        out = [c for c in out if c.get("framework") in only_framework or "baseline" in c]
    if only_model_tier:
        out = [c for c in out if c.get("model_tier") in only_model_tier or "baseline" in c]
    return out


def run_full_grid(
    use_mock: bool = False,
    out_path: str = "results/results.json",
    grid: list[dict] | None = None,
) -> list[dict]:
    # Same demand series reused across all configs within a metric, so
    # comparisons are apples-to-apples (paper's Appendix 6: one fixed
    # customer demand series per run). Deterministic (fixed seed), so this
    # is identical across separate shard processes without needing to share
    # any state between them.
    demand = generate_mjd_demand(FIXED_PARAMS["num_steps"], MJDParams())

    all_results = []
    for config in (grid if grid is not None else build_full_grid()):
        print(f"Running: {config['label']} ({config['metric']})...", flush=True)
        result = run_single_experiment(config, use_mock=use_mock, demand=demand)
        print(f"  -> cost={result['cost']:.1f}, bullwhip={result['bullwhip']:.5f}, "
              f"time={result['elapsed_sec']:.1f}s", flush=True)
        all_results.append(result)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def merge_result_files(paths: list[str], out_path: str) -> list[dict]:
    """Combine multiple shard result JSON files into one, for final reporting."""
    merged = []
    for p in paths:
        with open(p) as f:
            merged.extend(json.load(f))
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    return merged

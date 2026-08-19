"""
Format experiment results into tables matching the paper's layout, so
replicated numbers can be compared line-by-line against Table 1 (main text,
Section 6.1) and Table 2 (Section 6.2 / Appendix 4), rather than a table of
our own devising.
"""
from .config import MODELS

FRAMEWORK_DISPLAY_NAMES = {
    "standalone": "Standalone LLM",
    "info_sharing": "LLM with Info Sharing",
    "standalone_tool": "Standalone LLM + Tool",
    "info_sharing_tool": "Info Sharing + Tool",
    "negotiation_tool": "Negotiation + Tool",
}

MODEL_DISPLAY_NAMES = {
    "small": "Llama-3.1-8B (Flash-analog)",
    "large": "Llama-3.1-70B (Pro-analog)",
}

LABEL_WIDTH = 46


def _row_label(config: dict) -> str:
    if "baseline" in config:
        return config["label"]
    fw = FRAMEWORK_DISPLAY_NAMES.get(config["framework"], config["framework"])
    model = MODEL_DISPLAY_NAMES.get(config["model_tier"], config["model_tier"])
    return f"{fw} [{model}]"


def format_table(results: list[dict], metric: str) -> str:
    rows = [r for r in results if r["config"]["metric"] == metric]
    lines = [
        f"{'Configuration':<{LABEL_WIDTH}} {'Cost':>14} {'Bullwhip (CoV)':>16}",
        "-" * (LABEL_WIDTH + 32),
    ]
    for r in rows:
        label = _row_label(r["config"])
        if len(label) > LABEL_WIDTH:
            lines.append(label)
            label = ""
        lines.append(f"{label:<{LABEL_WIDTH}} {r['cost']:>14,.1f} {r['bullwhip']:>16.5f}")
    return "\n".join(lines)


def format_all_tables(results: list[dict]) -> str:
    out = []
    out.append("=" * 87)
    out.append("TABLE 1-STYLE: Global Cost Minimisation experiments")
    out.append("(compare against paper's Table 1 main text + Appendix 4 Table 1)")
    out.append("=" * 87)
    out.append(format_table(results, "cost"))
    out.append("")
    out.append("=" * 87)
    out.append("TABLE 2-STYLE: Global Bullwhip Effect experiments")
    out.append("(compare against paper's Table 2, Section 6.2)")
    out.append("=" * 87)
    out.append(format_table(results, "bullwhip"))
    return "\n".join(out)

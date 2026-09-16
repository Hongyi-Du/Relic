"""Build the public paper-result snapshot from a reviewed declarative source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from relic.paths import project_root


def default_source() -> Path:
    return project_root() / "reproduction" / "main_results" / "paper_results_source.yaml"


def default_output_directory() -> Path:
    return project_root() / "artifacts" / "paper_results"


def load_source(path: Path | None = None) -> dict[str, Any]:
    source = path or default_source()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"paper results source must be a mapping: {source}")
    validate(payload)
    return payload


def validate(payload: dict[str, Any]) -> None:
    main = payload["main_study"]
    if int(main["runs"]) != 240:
        raise ValueError("the canonical main study contains 240 runs")
    if sum(int(model["runs"]) for model in main["models"]) != 240:
        raise ValueError("main-study model counts must sum to 240")
    if list(main["arms"]) != ["B0", "B1", "B2", "B3"]:
        raise ValueError("canonical arm order must be B0, B1, B2, B3")
    if list(main["seeds"]) != [1401, 2711, 4013]:
        raise ValueError("canonical seeds do not match the paper")
    workload_ids = [row["id"] for row in main["workload_inventory"]]
    if workload_ids != [f"W{number:02d}" for number in range(1, 11)]:
        raise ValueError("canonical workload inventory must contain W01-W10 in paper order")
    for metric in main["metrics"]:
        if list(metric["arms"]) != ["B0", "B1", "B2", "B3"]:
            raise ValueError(f"metric {metric['id']} does not contain the canonical arm order")
    programbench = payload["external_extensions"]["programbench"]
    if programbench["reproduction_artifacts_included"] is not False:
        raise ValueError("ProgramBench reproduction assets are outside the first release")


def public_payload(source_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "relic-paper-results-v1",
        "paper": source_payload["paper"],
        "main_study": source_payload["main_study"],
        "protocol_census": source_payload["protocol_census"],
        "internal_transfer": source_payload["internal_transfer"],
        "binding_ablation": source_payload["binding_ablation"],
        "external_extensions": source_payload["external_extensions"],
        "hci": source_payload["hci"],
        "limitations": source_payload["limitations"],
    }


def serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _format_value(value: float, unit: str) -> str:
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "million_tokens":
        return f"{value:.3f}M"
    if unit == "thousand_tokens":
        return f"{value:g}k"
    if unit == "percentage_points":
        return f"{value:+g} pp"
    return str(value)


def _format_interval(interval: list[float], unit: str) -> str:
    low, high = interval
    if unit == "percent":
        return f"[{low:g}%, {high:g}%]"
    if unit == "million_tokens":
        return f"[{low:g}M, {high:g}M]"
    if unit == "thousand_tokens":
        return f"[{low:g}k, {high:g}k]"
    if unit == "percentage_points":
        return f"[{low:+g}, {high:+g}] pp"
    return f"[{low:g}, {high:g}]"


def _format_estimate(record: dict[str, Any], unit: str) -> str:
    value = _format_value(record["value"], unit)
    interval = record.get("ci95")
    if interval is None:
        return value
    return f"{value}<br>{_format_interval(interval, unit)}"


def render_markdown(payload: dict[str, Any]) -> str:
    paper = payload["paper"]
    main = payload["main_study"]
    lines = [
        "# Paper results",
        "",
        f"Canonical aggregate snapshot for *{paper['title']}* ({paper['status']}).",
        "",
        (
            "These values are transcribed from the paper, not recomputed from historical raw "
            "runs. The JSON artifact and its reviewed YAML source are the complete canonical "
            "snapshot; this page is a human-readable rendering of that same data."
        ),
        "",
        "## Main study",
        "",
        (
            f"{main['runs']} runs: 2 models × {main['workloads']} workloads × 3 seeds × 4 arms, "
            f"{main['ticks']} ticks per run, checkpoints every {main['checkpoint_every']} ticks, "
            f"and {main['sprint_ticks']}-tick sprints. Work rhythm is disabled; realized provider "
            "tokens are measured without a matched hard budget."
        ),
        "",
        "| Metric | B0 | B1 | B2 | B3 | B3-B2 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in main["metrics"]:
        unit = metric["unit"]
        arm_values = [_format_estimate(metric["arms"][arm], unit) for arm in main["arms"]]
        contrast = metric["b3_minus_b2"]
        contrast_value = _format_value(contrast["value"], contrast["unit"])
        lines.append(
            f"| {metric['label']} | {' | '.join(arm_values)} | "
            f"{contrast_value}<br>{_format_interval(contrast['ci95'], contrast['unit'])} |"
        )
    census = payload["protocol_census"]
    transfer = payload["internal_transfer"]
    ablation = payload["binding_ablation"]
    cooper = payload["external_extensions"]["cooperbench"]
    programbench = payload["external_extensions"]["programbench"]
    lines.extend(
        [
            "",
            "## Protocol census",
            "",
            (
                f"Across {census['b3_runs']} B3 runs: "
                f"{census['autonomous_proposal_lineages']} autonomous proposal lineages, "
                f"{census['currently_adopted_lineages']} adopted at endpoint, "
                f"{census['weak_or_strong_formed_lineages']} weak-or-strong formed "
                f"({census['weak_only_formed_lineages']} weak-only and "
                f"{census['strong_subset']} in the strong subset). Review/merge, release "
                "engineering, and evidence governance account for "
                f"{census['top_three_share_of_formed'] * 100:.1f}% "
                "of formed lineages."
            ),
            "",
            "## Internal transfer and binding ablation",
            "",
            "| Internal-transfer metric | Fresh | Text | Exec | Exec−Fresh (95% CI) | Exec−Text (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
            (
                "| Behavioral-case pass rate | "
                f"{_format_estimate(transfer['arms']['Fresh'], 'percent')} | "
                f"{_format_estimate(transfer['arms']['Text'], 'percent')} | "
                f"{_format_estimate(transfer['arms']['Exec'], 'percent')} | "
                f"{_format_estimate(transfer['contrasts']['exec_minus_fresh'], 'percentage_points')} | "
                f"{_format_estimate(transfer['contrasts']['exec_minus_text'], 'percentage_points')} |"
            ),
        ]
    )
    for metric in transfer["cost_metrics"]:
        unit = metric["unit"]
        lines.append(
            f"| {metric['label']} | "
            f"{_format_estimate(metric['arms']['Fresh'], unit)} | "
            f"{_format_estimate(metric['arms']['Text'], unit)} | "
            f"{_format_estimate(metric['arms']['Exec'], unit)} | "
            f"{_format_estimate(metric['contrasts']['exec_minus_fresh'], unit)} | "
            f"{_format_estimate(metric['contrasts']['exec_minus_text'], unit)} |"
        )
    lines.extend(
        [
            "",
            transfer["cost_metrics"][0]["scope"],
            "",
            (
                "Complete contracts in the in-situ binding ablation: B3-text "
                f"{ablation['b3_text_percent']}%, executable B3 "
                f"{ablation['executable_b3_percent']}%; difference "
                f"+{ablation['executable_minus_text']['value']} pp "
                f"(95% CI {ablation['executable_minus_text']['ci95']})."
            ),
            "",
            "Descriptive binding-ablation endpoints (B3-text → executable B3): "
            + "; ".join(
                f"{name.replace('_', ' ')} "
                f"{row['b3_text_percent']:.2f}% → {row['executable_b3_percent']:.2f}% "
                f"({row['executable_minus_text_percentage_points']:+.2f} pp)"
                for name, row in ablation["descriptive_endpoints"].items()
            )
            + ". Only complete contracts has the prespecified paired interval; it uses "
            f"{ablation['executable_minus_text']['bootstrap_replicates']:,} bootstrap draws.",
            "",
            "## External extensions",
            "",
            "CooperBench fixed 48-pair subset:",
            "",
        ]
    )
    for row in cooper["results"]:
        suffix = f"; {row['note']}" if row.get("note") else ""
        result = row.get("supported_range") or f"{row['successes']}/{row['total']}"
        lines.append(f"- {row['system']} ({row['model']}): {result}{suffix}")
    lines.extend(
        [
            "",
            (
                f"ProgramBench (same {programbench['tasks']} tasks): official mini-SWE-agent "
                f"{programbench['official_mini_swe_agent_percent']}%; with executable protocols "
                f"{programbench['with_executable_protocols_percent']}% "
                f"({programbench['absolute_gain_percentage_points']:+.3f} pp; "
                f"{programbench['relative_gain_percent']:+.1f}% relative). Both have 2/25 tasks at "
                "or above 95%. SDL is disabled: this is a protocol plug-in result in another "
                "harness, separate from the main study and internal transfer."
            ),
            "",
            "ProgramBench reproduction code and artifacts are not included in this release.",
            "",
            "## Interpretation boundaries",
            "",
            (
                "Evaluator provenance is retained per run. Formal evaluation requires an untrusted, "
                "network-disabled Linux/amd64 container from a digest-pinned image; the paper does "
                "not claim one fixed backend or evaluator-environment hash for the full matrix."
            ),
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append(f"- HCI: {payload['hci']['claim_boundary']}")
    lines.append(f"- CooperBench: {cooper['claim_boundary']}")
    return "\n".join(lines) + "\n"


def write_results(source: Path | None = None, output_directory: Path | None = None) -> tuple[Path, Path]:
    payload = public_payload(load_source(source))
    destination = output_directory or default_output_directory()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "paper_results.json"
    markdown_path = destination / "paper_results.md"
    json_path.write_text(serialize_json(payload), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path

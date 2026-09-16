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
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    paper = payload["paper"]
    main = payload["main_study"]
    lines = [
        "# Paper results",
        "",
        f"Canonical aggregate snapshot for *{paper['title']}* ({paper['status']}).",
        "",
        "These values are transcribed from the paper, not recomputed from historical raw runs.",
        "",
        "## Main study",
        "",
        (
            f"{main['runs']} runs: 2 models × {main['workloads']} workloads × 3 seeds × 4 arms, "
            f"{main['ticks']} ticks per run."
        ),
        "",
        "| Metric | B0 | B1 | B2 | B3 | B3-B2 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in main["metrics"]:
        unit = metric["unit"]
        arm_values = [_format_value(metric["arms"][arm]["value"], unit) for arm in main["arms"]]
        contrast = metric["b3_minus_b2"]
        contrast_value = _format_value(contrast["value"], contrast["unit"])
        ci = contrast["ci95"]
        lines.append(
            f"| {metric['label']} | {' | '.join(arm_values)} | "
            f"{contrast_value} [{ci[0]}, {ci[1]}] |"
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
                f"{census['adopted_at_endpoint']} adopted at endpoint, "
                f"{census['sustained_use']} sustained-use, and "
                f"{census['stronger_execution_or_outcome_evidence']} with stronger "
                "execution/outcome evidence."
            ),
            "",
            "## Internal transfer and binding ablation",
            "",
            (
                f"Behavioral-case pass: Fresh {transfer['arms']['Fresh']['value']}%, "
                f"Text {transfer['arms']['Text']['value']}%, "
                f"Exec {transfer['arms']['Exec']['value']}%. Exec-Text is "
                f"+{transfer['contrasts']['exec_minus_text']['value']} pp "
                f"(95% CI {transfer['contrasts']['exec_minus_text']['ci95']})."
            ),
            "",
            (
                "Complete contracts in the in-situ binding ablation: B3-text "
                f"{ablation['b3_text_percent']}%, executable B3 "
                f"{ablation['executable_b3_percent']}%; difference "
                f"+{ablation['executable_minus_text']['value']} pp "
                f"(95% CI {ablation['executable_minus_text']['ci95']})."
            ),
            "",
            "## External extensions",
            "",
            "CooperBench fixed 48-pair subset:",
            "",
        ]
    )
    for row in cooper["results"]:
        lines.append(f"- {row['system']} ({row['model']}): {row['successes']}/{row['total']}")
    lines.extend(
        [
            "",
            (
                f"ProgramBench (same {programbench['tasks']} tasks): official mini-SWE-agent "
                f"{programbench['official_mini_swe_agent_percent']}%; with executable protocols "
                f"{programbench['with_executable_protocols_percent']}%. Both have 2/25 tasks at "
                "or above 95%."
            ),
            "",
            "ProgramBench reproduction code and artifacts are not included in this release.",
            "",
            "## Interpretation boundaries",
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

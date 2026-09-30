"""Historical product cases for external-society validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from .schemas import Artifact


@dataclass(frozen=True)
class HistoricalProductCase:
    case_id: str
    product_name: str
    initial_release_date: date
    comparison_release_date: date
    real_elapsed_days: int
    source_urls: tuple[str, ...]
    agent_visible_source_urls: tuple[str, ...]
    initial_artifact: Artifact
    target_improvement_themes: tuple[str, ...]


def vite_2020_case() -> HistoricalProductCase:
    initial_release = date(2020, 6, 19)
    comparison_release = date(2021, 2, 16)
    return HistoricalProductCase(
        case_id="vite_2020_initial_to_v2",
        product_name="Vite",
        initial_release_date=initial_release,
        comparison_release_date=comparison_release,
        real_elapsed_days=(comparison_release - initial_release).days,
        source_urls=(
            "https://github.com/vitejs/vite",
            "https://github.com/vitejs/vite/commit/ec78097539d03ce405579a91a9f329fdd14f5842",
            "https://github.com/vitejs/vite/commit/0deadcde614fee4fe2a5ccc9f5321dc30bfcca2f",
            "https://registry.npmjs.org/vite/-/vite-0.20.10.tgz",
            "https://registry.npmjs.org/vite/-/vite-2.0.0.tgz",
        ),
        agent_visible_source_urls=(
            "https://github.com/vitejs/vite",
            "https://github.com/vitejs/vite/commit/ec78097539d03ce405579a91a9f329fdd14f5842",
            "https://registry.npmjs.org/vite/-/vite-0.20.10.tgz",
        ),
        initial_artifact=Artifact(
            id="vite_early_dev_server",
            provider_id="vite_team",
            artifact_kind="developer_tool",
            capability_profile={
                "dev_server_speed": 0.86,
                "hmr_feedback_loop": 0.78,
                "production_build": 0.48,
                "framework_generalization": 0.32,
                "plugin_ecosystem": 0.26,
                "browser_compatibility": 0.36,
            },
            reliability_profile={
                "dev_server_stability": 0.66,
                "css_build_reliability": 0.42,
                "dependency_resolution": 0.44,
                "cross_framework_reliability": 0.30,
            },
            cost_profile={"money": 0.0, "migration_time": 0.34, "learning_time": 0.28},
            access_constraints={"node_tooling_familiarity": 0.35},
            learning_curve=0.44,
            failure_modes=(
                "production_build_edge_case",
                "css_or_asset_pipeline_surprise",
                "plugin_missing",
                "framework_mismatch",
                "dependency_resolution_failure",
            ),
            evidence_claims=(
                "fast dev server",
                "hot module replacement",
                "modern frontend workflow",
            ),
            version="0.20.10",
            release_time=0,
            public_claims=(
                "Fast frontend development server for modern web projects.",
                "Lower feedback-loop latency than traditional bundler workflows.",
            ),
            task_fit_distribution={
                "cold_start_vue_app": 0.84,
                "hmr_component_edit": 0.82,
                "production_build_css_assets": 0.43,
                "plugin_integration": 0.30,
                "non_vue_framework_setup": 0.28,
                "legacy_browser_output": 0.34,
            },
        ),
        target_improvement_themes=(
            "framework_generalization",
            "plugin_ecosystem",
            "production_build_reliability",
            "dependency_resolution",
            "documentation_and_migration_path",
            "legacy_browser_compatibility",
        ),
    )


def snowpack_2020_case() -> HistoricalProductCase:
    initial_release = date(2020, 5, 27)
    comparison_release = date(2021, 1, 12)
    return HistoricalProductCase(
        case_id="snowpack_2020_v2_to_v3",
        product_name="Snowpack",
        initial_release_date=initial_release,
        comparison_release_date=comparison_release,
        real_elapsed_days=(comparison_release - initial_release).days,
        source_urls=(
            "https://github.com/FredKSchott/snowpack",
            "https://www.snowpack.dev/posts/2020-05-26-snowpack-2-0-release/",
            "https://registry.npmjs.org/snowpack/-/snowpack-2.0.0.tgz",
            "https://registry.npmjs.org/snowpack/-/snowpack-3.0.1.tgz",
        ),
        agent_visible_source_urls=(
            "https://github.com/FredKSchott/snowpack",
            "https://www.snowpack.dev/posts/2020-05-26-snowpack-2-0-release/",
            "https://registry.npmjs.org/snowpack/-/snowpack-2.0.0.tgz",
        ),
        initial_artifact=Artifact(
            id="snowpack_v2_unbundled_toolchain",
            provider_id="snowpack_team",
            artifact_kind="developer_tool",
            capability_profile={
                "dev_server_speed": 0.84,
                "hmr_feedback_loop": 0.58,
                "production_build": 0.46,
                "plugin_ecosystem": 0.42,
                "framework_generalization": 0.36,
                "ssr_runtime": 0.12,
                "source_provider_architecture": 0.22,
            },
            reliability_profile={
                "dev_server_stability": 0.58,
                "build_pipeline_reliability": 0.43,
                "dependency_resolution": 0.40,
                "hmr_error_recovery": 0.34,
                "framework_adapter_reliability": 0.32,
            },
            cost_profile={"money": 0.0, "migration_time": 0.42, "learning_time": 0.36},
            access_constraints={"node_tooling_familiarity": 0.40},
            learning_curve=0.50,
            failure_modes=(
                "dependency_resolution_failure",
                "runtime_or_build_failure",
                "hmr_error_dead_end",
                "plugin_missing",
                "framework_adapter_gap",
                "source_provider_gap",
                "ssr_runtime_missing",
            ),
            evidence_claims=(
                "unbundled development",
                "fast dev server",
                "production build scripts",
                "third-party plugins",
            ),
            version="2.0.0",
            release_time=0,
            public_claims=(
                "Modern unbundled development with fast rebuilds.",
                "Production builds can add back bundlers and optimization.",
            ),
            task_fit_distribution={
                "dev_server_start": 0.82,
                "production_build": 0.42,
                "framework_integration": 0.36,
                "plugin_extension": 0.40,
                "dependency_upgrade": 0.38,
                "team_workflow": 0.44,
                "documentation_lookup": 0.50,
                "legacy_browser_support": 0.28,
                "ssr_runtime": 0.12,
                "source_provider_integration": 0.20,
            },
        ),
        target_improvement_themes=(
            "architecture_generalization",
            "plugin_ecosystem",
            "framework_generalization",
            "production_build_reliability",
            "dependency_resolution",
            "diagnostics_and_recovery",
            "documentation_and_migration_path",
        ),
    )


def gitingest_2025_case() -> HistoricalProductCase:
    initial_release = date(2025, 7, 2)
    comparison_release = date(2025, 7, 30)
    return HistoricalProductCase(
        case_id="gitingest_2025_v015_to_v020",
        product_name="gitingest",
        initial_release_date=initial_release,
        comparison_release_date=comparison_release,
        real_elapsed_days=(comparison_release - initial_release).days,
        source_urls=(
            "https://github.com/coderamp-labs/gitingest",
            "https://github.com/coderamp-labs/gitingest/tree/v0.1.5",
            "https://github.com/coderamp-labs/gitingest/tree/v0.2.0",
            "https://github.com/coderamp-labs/gitingest/pull/437",
            "https://github.com/coderamp-labs/gitingest/pull/313",
            "https://github.com/coderamp-labs/gitingest/pull/464",
            "https://github.com/coderamp-labs/gitingest/pull/416",
            "https://github.com/coderamp-labs/gitingest/pull/388",
        ),
        agent_visible_source_urls=(
            "https://github.com/coderamp-labs/gitingest",
            "https://github.com/coderamp-labs/gitingest/tree/v0.1.5",
        ),
        initial_artifact=Artifact(
            id="gitingest_v015_repo_digest",
            provider_id="coderamp_labs",
            artifact_kind="developer_tool",
            capability_profile={
                "repository_digest_generation": 0.78,
                "local_directory_ingestion": 0.74,
                "token_estimation": 0.46,
                "pattern_filtering": 0.48,
                "gitignore_support": 0.42,
                "submodule_support": 0.12,
                "http_portability": 0.35,
                "large_file_controls": 0.30,
            },
            reliability_profile={
                "offline_token_count_resilience": 0.28,
                "max_file_size_enforcement": 0.30,
                "include_exclude_consistency": 0.46,
                "gitignore_processing": 0.44,
                "remote_repo_reachability": 0.38,
                "public_api_stability": 0.50,
            },
            cost_profile={"money": 0.0, "setup_time": 0.24, "learning_time": 0.22},
            access_constraints={"python_tooling_familiarity": 0.28},
            learning_curve=0.32,
            failure_modes=(
                "token_count_network_failure",
                "max_file_size_leak",
                "include_exclude_pattern_mismatch",
                "gitignore_or_gitingestignore_leak",
                "submodule_content_missing",
                "curl_binary_dependency",
                "path_traversal_risk",
            ),
            evidence_claims=(
                "turn a repository into an LLM-friendly text digest",
                "directory tree plus file contents",
                "token estimate for prompt budgeting",
            ),
            version="0.1.5",
            release_time=0,
            public_claims=(
                "Generate prompt-friendly repository digests for LLM workflows.",
                "Support local paths and GitHub repositories from a CLI or library API.",
            ),
            task_fit_distribution={
                "local_repo_digest": 0.80,
                "offline_digest_generation": 0.34,
                "large_repository_digest": 0.38,
                "include_exclude_filtering": 0.44,
                "gitignore_respecting_digest": 0.42,
                "submodule_repository_digest": 0.14,
                "remote_repo_check": 0.36,
                "security_review_path_handling": 0.32,
            },
        ),
        target_improvement_themes=(
            "token_count_resilience",
            "max_file_size_enforcement",
            "include_submodules",
            "ignore_pattern_reliability",
            "pattern_filtering_consistency",
            "http_client_portability",
            "diagnostics_and_recovery",
            "path_security",
        ),
    )


def gitingest_2025_v015_to_v030_case() -> HistoricalProductCase:
    """Corrected full-tag time machine spanning v0.1.5 through v0.3.0."""

    initial_release = date(2025, 6, 25)
    comparison_release = date(2025, 7, 30)
    base = gitingest_2025_case()
    return replace(
        base,
        case_id="gitingest_2025_v015_to_v030",
        initial_release_date=initial_release,
        comparison_release_date=comparison_release,
        real_elapsed_days=(comparison_release - initial_release).days,
        source_urls=(
            "https://github.com/coderamp-labs/gitingest",
            "https://github.com/coderamp-labs/gitingest/tree/v0.1.5",
            "https://github.com/coderamp-labs/gitingest/tree/v0.3.0",
            *tuple(url for url in base.source_urls if "/pull/" in url),
        ),
    )

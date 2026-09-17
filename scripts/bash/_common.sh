#!/usr/bin/env bash
# Shared, intentionally small helpers for the public Bash launchers.

set -euo pipefail

relic_load_repo_env() {
    local repo_root="$1"
    local env_file="${repo_root}/.env"
    local line key value

    [[ -f "$env_file" ]] || return 0

    # Parse dotenv-style assignments without sourcing the file. A .env file may
    # contain provider credentials, so it must neither be printed nor executed
    # as shell code.
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" || "${line:0:1}" == "#" ]] && continue

        if [[ "$line" =~ ^export[[:space:]]+ ]]; then
            line="${line#export}"
            line="${line#"${line%%[![:space:]]*}"}"
        fi

        if [[ "$line" != *=* ]]; then
            printf 'Invalid .env entry in %s\n' "$env_file" >&2
            return 2
        fi

        key="${line%%=*}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${line#*=}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"

        if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            printf 'Invalid .env variable name in %s\n' "$env_file" >&2
            return 2
        fi

        # Loading arbitrary process-control variables (PATH, LD_PRELOAD,
        # BASH_ENV, PYTHONPATH, and similar) would turn dotenv parsing into an
        # indirect execution primitive. Keep this list aligned with the public
        # environment contract instead.
        case "$key" in
            OPENAI_API_KEY | OPENAI_BASE_URL | OPENAI_MODEL | \
            RELIC_OPENAI_DEFAULT_HEADERS_JSON | RELIC_OPENAI_DISABLE_RESPONSE_STORAGE | \
            RELIC_EVALUATOR_MODE | RELIC_EVALUATOR_STRICT_REPRODUCIBILITY | \
            RELIC_EVALUATOR_BACKEND | RELIC_EVALUATOR_CONTAINER_IMAGE | \
            RELIC_EVALUATOR_CONTAINER_PLATFORM | RELIC_RUNTIME_MODEL | \
            ORG_LLM_RUNTIME_MODEL | RELIC_OUTPUT_ROOT | \
            RELIC_BENCHMARK_ROOT | RELIC_CACHE_ROOT | RELIC_INSPECTOR_PORT | \
            RELIC_UID | RELIC_GID | RELIC_TRACE_FILE | \
            RELIC_CLAUDE_OPUS_4_6_MODEL | \
            ORG_OSS_QUALIFICATION_TIMEOUT | ORG_LLM_API_KEY | ORG_LLM_BASE_URL | \
            ORG_LLM_WIRE_API | APPTAINER_CACHEDIR | APPTAINER_TMPDIR)
                ;;
            *)
                printf 'Unsupported variable in %s: %s\n' "$env_file" "$key" >&2
                return 2
                ;;
        esac

        if [[ "$value" == \"*\" && "$value" == *\" && ${#value} -ge 2 ]] || \
            [[ "$value" == \'*\' && "$value" == *\' && ${#value} -ge 2 ]]; then
            value="${value:1:${#value}-2}"
        fi

        export "$key=$value"
    done < "$env_file"
}

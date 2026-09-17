#!/usr/bin/env bash

set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SCRIPT_SOURCE" ]]; do
    SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)"
    SCRIPT_TARGET="$(readlink -- "$SCRIPT_SOURCE")"
    if [[ "$SCRIPT_TARGET" == /* ]]; then
        SCRIPT_SOURCE="$SCRIPT_TARGET"
    else
        SCRIPT_SOURCE="${SCRIPT_DIR}/${SCRIPT_TARGET}"
    fi
done
SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
relic_load_repo_env "$REPO_ROOT"
cd -- "$REPO_ROOT"

# The CLI owns selection validation, upstream-boundary checks, resume identity,
# and execution. This wrapper deliberately does not infer external paths,
# credentials, task images, or model aliases.
exec uv run relic run-cooper "$@"

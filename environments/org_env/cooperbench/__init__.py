"""Two-member B3 adapter for CooperBench.

The integration deliberately lives behind CooperBench's external-agent seam.
Importing :mod:`environments.org_env.cooperbench` does not require CooperBench
itself; only ``adapter`` imports its optional registration API.
"""

from environments.org_env.cooperbench.contract import (
    CONTRACT_SCHEMA_VERSION,
    TREATMENT_ID,
    FeatureCall,
    PairOutcome,
    PairRequest,
)

__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "TREATMENT_ID",
    "FeatureCall",
    "PairOutcome",
    "PairRequest",
]

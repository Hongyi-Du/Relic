"""lived.core — foundational types & math (leaf modules).

contracts (dataclasses) · schema (feature/trait/event specs) · ports (extension
points) · transforms · wmatrix (profile→weights) · actions (action vocabulary).
These have no dependencies on the rest of ``lived`` (or only on each other), so
everything else builds on top. Public symbols are re-exported from
``agent_sdk.lived`` — import from there, or from the concrete submodule.
"""

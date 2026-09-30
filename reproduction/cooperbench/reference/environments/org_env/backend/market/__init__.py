"""OrgEnv external market (v14 P5).

The post-release market-validation loop: once a release is *published*, simulated
external users run trials against it, leave customer tickets / feedback, and signal
willingness-to-pay. Crucially, trial satisfaction + WTP track the product's REAL
runtime quality (does it run end-to-end and produce grounded output?), so the
``customers`` funding milestone can only be won by actually shipping a usable
product — not by churning paperwork or gaming release gates.
"""
from environments.org_env.backend.market.validation import (
    product_quality, run_market_trials, market_summary,
)

__all__ = ["product_quality", "run_market_trials", "market_summary"]

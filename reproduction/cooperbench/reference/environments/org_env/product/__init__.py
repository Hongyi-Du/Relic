"""Messy product substrate — the concrete research-agent prototype the team works on."""
from environments.org_env.product.objects import ProductArtifact, ProductState
from environments.org_env.product.seed import DEFAULT_COMPANY_CONFIG, seed_product

__all__ = ["ProductState", "ProductArtifact", "seed_product", "DEFAULT_COMPANY_CONFIG"]

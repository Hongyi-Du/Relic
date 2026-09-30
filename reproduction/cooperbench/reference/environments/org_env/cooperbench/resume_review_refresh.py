"""Route obsolete rejected checkpoint reviews back to their actual peer."""
from copy import deepcopy
from collections.abc import Mapping


def refresh_legacy_rejections(world):
    from .semantic_review import PROBE_VALIDITY_SCHEMA, pull_request_revision
    from environments.org_env.backend.repo.repo import PRStatus
    reviews = getattr(world, '_cooperbench_semantic_reviews', {})
    owners = (getattr(world, '_cooperbench_sdl_state', {}) or {}).get('feature_owners', {})
    repo = getattr(getattr(world, 'repo_system', None), 'repo', None)
    prs = getattr(repo, 'pull_requests', {}) or {}
    refreshed = []
    for feature, receipt in list(reviews.items()):
        if not isinstance(receipt, Mapping) or receipt.get('approved') is not False:
            continue
        validity = receipt.get('probe_validity') or {}
        schema = validity.get('schema_version') if isinstance(validity, Mapping) else None
        if schema not in {'cooperbench_probe_failure_validity_v5', 'cooperbench_probe_failure_validity_v6',
                          'cooperbench_probe_failure_validity_v7_fixture_origin',
                          'cooperbench_probe_failure_validity_v8_decoded_syntax',
                          'cooperbench_probe_failure_validity_v9_observer_preconditions',
                          'cooperbench_probe_failure_validity_v10_input_protocols'}:
            continue
        pr = prs.get(receipt.get('pr_id'))
        author = owners.get(feature)
        if (pr is None or not author or getattr(pr, 'author_id', None) != author
                or str(getattr(pr.status, 'value', pr.status)) != 'changes_requested'
                or receipt.get('reviewed_pr_revision') != pull_request_revision(pr)
                or receipt.get('reviewer_id') not in set(owners.values()) - {author}):
            continue
        # This removes obsolete repair authority, not a failed execution or a
        # product requirement. The unchanged PR needs a new real peer verdict.
        world.__dict__.setdefault('_cooperbench_retired_checkpoint_rejections', []).append({
            'feature_id': feature, 'retired_at_tick': world.world_tick,
            'old_review': deepcopy(receipt), 'new_validity_schema': PROBE_VALIDITY_SCHEMA,
            'reason': 'focused_validity_policy_changed_requires_peer_revalidation',
        })
        del reviews[feature]
        failures = getattr(world, '_cooperbench_actor_patch_guard_failures', {})
        if isinstance(failures, dict) and author in failures:
            world._cooperbench_retired_checkpoint_rejections[-1]['old_guard'] = deepcopy(failures.pop(author))
        pr.status = PRStatus.REVIEW_REQUESTED
        refreshed.append(feature)
    return refreshed

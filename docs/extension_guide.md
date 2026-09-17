# Extension guide

Relic is primarily a paper artifact. Extensions should preserve the canonical
main-study contract and make their scope explicit rather than silently changing
the paper executor.

## Choose the correct branch

- General core bug fix: make it in `main` first, then merge it into extension
  branches.
- HCI-only code, configs, replay assets, or tests: use `hci`.
- CooperBench adapter, upstream compatibility material, or Cooper-specific
  tests: use `cooper`.
- Additional sanitized core regression coverage: keep it in the ordinary
  repository test suite.

Do not put ProgramBench assets or entrypoints in any public branch. Do not turn
an extension branch into a divergent copy of the core runtime.

## Adding a public configuration or command

1. Keep execution logic in the Python CLI.
2. Add Bash and WSL-only PowerShell wrappers only as argument-forwarding
   conveniences; neither wrapper may duplicate experiment logic.
3. Document the configuration precedence, output directory, provider cost, and
   any required external data or credentials.
4. Add a release-focused test that is safe by default. Expensive, Docker, or
   provider tests must be opt-in markers.
5. Preserve the public/private trace boundary and avoid secrets, personal paths,
   historical raw data, or unreleasable fixtures.

## External benchmark integrations

If an upstream benchmark cannot be redistributed, ship only public adapter
code, task IDs/manifest when allowed, compatibility patches, and setup
instructions. Pin the upstream revision and license boundary; do not replace a
missing dataset, evaluator, task image, or historical result with a local
substitute. The `cooper` branch demonstrates this policy.

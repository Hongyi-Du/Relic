# Relic documentation

This is the maintained public documentation set for the paper artifact. It is
intentionally smaller than the private development history and records the
current release boundary rather than promising unavailable author assets.

| Document | Use it for |
| --- | --- |
| [Installation](installation.md) | Native Linux/WSL2 and Docker setup |
| [Environment and platform support](environment.md) | Supported platforms, variables, WSL2/PowerShell, Docker, outputs, concurrency |
| [Configuration](configuration.md) | Canonical configs, precedence, model/benchmark/output choices |
| [Architecture](architecture.md) | Public component boundaries and execution/data flow |
| [Protocol lifecycle](protocol_lifecycle.md) | What B3 records as proposal, adoption, use, enforcement, revision, and retirement |
| [Inspector](inspector.md) | `relic-trace-v1`, replay, privacy, and network boundary |
| [Transfer](transfer.md) | Final Text/Exec transfer design and its locked protocol bundle |
| [Evaluator](evaluator.md) | Local source closure versus the formal-paper binding gap |
| [Release scope](release-scope.md) | Branch topology, exclusions, and synchronization policy |
| [Source provenance](source-provenance.md) | Frozen source closure and narrow release adaptations |
| [Reproduction run report](REPRODUCTION_RUN_REPORT.md) | What was actually executed in the final local run |
| [Known release gaps](KNOWN_RELEASE_GAPS.md) | Author, legal, tag, remote, and external benchmark dependencies |

Paper-facing command and asset mapping lives in [`../reproduction/`](../reproduction/README.md).

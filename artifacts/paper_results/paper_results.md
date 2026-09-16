# Paper results

Canonical aggregate snapshot for *Relic: From Multi-Agent Collaboration to Organizational Capability* (ICLR 2027 under review).

These values are transcribed from the paper, not recomputed from historical raw runs.

## Main study

240 runs: 2 models × 10 workloads × 3 seeds × 4 arms, 336 ticks per run.

| Metric | B0 | B1 | B2 | B3 | B3-B2 (95% CI) |
|---|---:|---:|---:|---:|---:|
| Complete contracts on mainline | 10.30% | 18.21% | 16.31% | 22.82% | 6.51 [2.812, 10.517] |
| Held-out cases on mainline | 10.29% | 12.25% | 13.86% | 24.51% | 10.65 [2.041, 19.498] |
| Exposed cases on mainline | 14.89% | 26.12% | 22.57% | 30.30% | 7.73 [2.183, 13.016] |
| Evaluator-confirmed seeded issues | 13.35% | 22.97% | 20.84% | 28.76% | 7.92 [2.908, 14.409] |
| Average tokens per run | 3.522M | 43.903M | 3.751M | 6.523M | 2.772M [1.373, 4.665] |
| Tokens per evaluator-confirmed issue | 4.947M | 31.760M | 2.907M | 3.651M | -0.174M [-1.686, 0.633] |

## Protocol census

Across 60 B3 runs: 497 autonomous proposal lineages, 393 adopted at endpoint, 280 sustained-use, and 32 with stronger execution/outcome evidence.

## Internal transfer and binding ablation

Behavioral-case pass: Fresh 25.4%, Text 34.6%, Exec 41.2%. Exec-Text is +6.5 pp (95% CI [0.7, 15.6]).

Complete contracts in the in-situ binding ablation: B3-text 15.03%, executable B3 22.2%; difference +7.18 pp (95% CI [3.54, 10.91]).

## External extensions

CooperBench fixed 48-pair subset:

- Relic B3-2 (Claude Opus 4.6): 29/48
- Official Solo (Claude Opus 4.6): 26/48
- Official Peer (Claude Opus 4.6): 13/48
- Team with protocol verbs (GPT-5.5-hao): 26/48
- Team without protocol verbs (GPT-5.5-hao): 24/48

ProgramBench (same 25 tasks): official mini-SWE-agent 64.164%; with executable protocols 70.916%. Both have 2/25 tasks at or above 95%.

ProgramBench reproduction code and artifacts are not included in this release.

## Interpretation boundaries

- B3 does not improve every workload; W01 is negative and W05 is zero on the named contrast.
- ProgramBench has aggregate reporting only in this release.
- Historical per-run author data and complete raw trajectories are not included.
- HCI: The appendix reports design and formative observations, not powered participant-study evidence, population estimates, or significance tests.
- CooperBench: The 48 pairs are not 48 independent samples, the selected subset was used during adapter development, and no statistical-superiority claim is made.

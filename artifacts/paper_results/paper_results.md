# Paper results

Canonical aggregate snapshot for *Relic: From Multi-Agent Collaboration to Organizational Capability* (ICLR 2027 under review).

These values are transcribed from the paper, not recomputed from historical raw runs. The JSON artifact and its reviewed YAML source are the complete canonical snapshot; this page is a human-readable rendering of that same data.

## Main study

240 runs: 2 models × 10 workloads × 3 seeds × 4 arms, 336 ticks per run, checkpoints every 24 ticks, and 168-tick sprints. Work rhythm is disabled; realized provider tokens are measured without a matched hard budget.

| Metric | B0 | B1 | B2 | B3 | B3-B2 (95% CI) |
|---|---:|---:|---:|---:|---:|
| Complete contracts on mainline | 10.30%<br>[5.1%, 16.4%] | 18.21%<br>[10.9%, 26.6%] | 16.31%<br>[10.3%, 26.8%] | 22.82%<br>[14.6%, 32%] | +6.51 pp<br>[+2.812, +10.517] pp |
| Held-out cases on mainline | 10.29%<br>[0%, 35.5%] | 12.25%<br>[0%, 31.3%] | 13.86%<br>[1%, 46.1%] | 24.51%<br>[1.7%, 65.3%] | +10.65 pp<br>[+2.041, +19.498] pp |
| Exposed cases on mainline | 14.89%<br>[7.2%, 24.1%] | 26.12%<br>[14.9%, 39.3%] | 22.57%<br>[15%, 41.3%] | 30.30%<br>[17.9%, 44.1%] | +7.73 pp<br>[+2.183, +13.016] pp |
| Evaluator-confirmed seeded issues | 13.35%<br>[6.5%, 21.5%] | 22.97%<br>[13.1%, 34.6%] | 20.84%<br>[12.2%, 30.1%] | 28.76%<br>[17.4%, 41.1%] | +7.92 pp<br>[+2.908, +14.409] pp |
| Average tokens per run | 3.522M<br>[3.35M, 3.695M] | 43.903M<br>[40.745M, 47.262M] | 3.751M<br>[3.439M, 4.054M] | 6.523M<br>[5.119M, 8.379M] | 2.772M<br>[1.373M, 4.665M] |
| Tokens per evaluator-confirmed issue | 4.947M<br>[3.396M, 5.963M] | 31.760M<br>[25.994M, 46.592M] | 2.907M<br>[2.402M, 4.632M] | 3.651M<br>[2.359M, 6.66M] | -0.174M<br>[-1.686M, 0.633M] |

## Protocol census

Across 60 B3 runs: 497 autonomous proposal lineages, 393 adopted at endpoint, 280 weak-or-strong formed (248 weak-only and 32 in the strong subset). Review/merge, release engineering, and evidence governance account for 87.1% of formed lineages.

## Internal transfer and binding ablation

| Internal-transfer metric | Fresh | Text | Exec | Exec−Fresh (95% CI) | Exec−Text (95% CI) |
|---|---:|---:|---:|---:|---:|
| Behavioral-case pass rate | 25.40%<br>[13.2%, 41.1%] | 34.60%<br>[24.1%, 46.4%] | 41.20%<br>[23.4%, 54.5%] | +15.8 pp<br>[+9.2, +31.8] pp | +6.5 pp<br>[+0.7, +15.6] pp |
| Average tokens per run | 2.381M<br>[1.747M, 3.189M] | 2.615M<br>[2.213M, 3.033M] | 2.448M<br>[2.199M, 2.72M] | 0.067M<br>[-0.569M, 0.532M] | -0.167M<br>[-0.467M, 0.073M] |
| Tokens per evaluated case | 176k<br>[106k, 271k] | 154k<br>[96k, 265k] | 144k<br>[94k, 243k] | -32k<br>[-112k, 31k] | -10k<br>[-33k, 4k] |
| Tokens per verified pass | 691k<br>[488k, 1419k] | 444k<br>[271k, 935k] | 350k<br>[196k, 958k] | -342k<br>[-818k, -14k] | -94k<br>[-152k, 54k] |

Separate nine-workload comparison: Fresh seed 2711 and Text/Exec seed 1401, paired by workload; this is not a same-seed cost estimate.

Complete contracts in the in-situ binding ablation: B3-text 15.03%, executable B3 22.2%; difference +7.18 pp (95% CI [3.54, 10.91]).

Descriptive binding-ablation endpoints (B3-text → executable B3): held out cases 8.07% → 36.98% (+28.91 pp); exposed cases 18.47% → 32.52% (+14.04 pp); evaluator confirmed seeded issues 17.72% → 29.44% (+11.71 pp). Only complete contracts has the prespecified paired interval; it uses 200,000 bootstrap draws.

## External extensions

CooperBench fixed 48-pair subset:

- Relic B3-2 (Claude Opus 4.6): 29/48
- Official Solo (Claude Opus 4.6): 26/48
- Official Peer (Claude Opus 4.6): 13/48
- Team with protocol verbs (GPT-5.5-hao): 26/48
- Team without protocol verbs (GPT-5.5-hao): 24/48
- OpenHands Cooperative, No Git (GPT-5): 13–15/48; Current official cooperative leaderboard reference; not a same-model control.

ProgramBench (same 25 tasks): official mini-SWE-agent 64.164%; with executable protocols 70.916% (+6.752 pp; +10.5% relative). Both have 2/25 tasks at or above 95%. SDL is disabled: this is a protocol plug-in result in another harness, separate from the main study and internal transfer.

ProgramBench reproduction code and artifacts are not included in this release.

## Interpretation boundaries

Evaluator provenance is retained per run. Formal evaluation requires an untrusted, network-disabled Linux/amd64 container from a digest-pinned image; the paper does not claim one fixed backend or evaluator-environment hash for the full matrix.

- B3 does not improve every workload; W01 is negative and W05 is zero on the named contrast.
- ProgramBench has aggregate reporting only in this release.
- Historical per-run author data and complete raw trajectories are not included.
- HCI: The appendix reports design and formative observations, not powered participant-study evidence, population estimates, or significance tests.
- CooperBench: The 48 pairs are not 48 independent samples, the selected subset was used during adapter development, and no statistical-superiority claim is made.

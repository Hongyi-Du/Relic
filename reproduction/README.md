# Reproduction map

This directory maps public release entrypoints to paper-facing material. It
does not contain reconstructed historical raw runs, provider transcripts, or
private states.

| Area | Public inputs and entrypoint | Output / current boundary |
| --- | --- | --- |
| Main results | [`main_results/`](main_results/) and `relic run-main` | Source-backed 120-cell plan and user-created output; formal runs require author evaluator bindings |
| Transfer | [`transfer/`](transfer/) and `relic run-transfer` | 60 fresh-B2 Text/Exec targets; formal execution has the same binding boundary |
| HCI | [`hci/`](hci/) on `main`; the implementation is on branch `hci` | Formative interface/replay scope; no selected historical trace is fabricated |
| Case study | [`case_study/`](case_study/) | Reserved for author-supplied sanitized selected traces and provenance |
| CooperBench | `cooper` branch only | See `reproduction/cooperbench/README.md` after switching branches |

`artifacts/paper_results/` is the only canonical public aggregate snapshot.
It is transcribed from the final paper rather than regenerated from unavailable
historical raw data.

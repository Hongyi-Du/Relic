# LanternForge — a scrubbable view of one OrgEnv run

An interactive visualization of the **cattrs B3 seed 4013** run: eight agents at a fictional
company work a real upstream migration for 336 simulated hours, and you can scrub the whole thing
hour by hour or watch it told as a five-act story.

Plain static files — no build step, no dependencies, no network access. Everything it needs is in
this directory.

```
demo/
  index.html           the page
  app.js               clock, office rendering, split screen
  acts.js              the five-act narrative
  workbench.js         the work / governance / channels panels
  i18n.js              all narration, English and Chinese
  scene.js sprites.js  pixel-art office and agents
  styles.css
  data/                the two generated JSON files the page loads
  build/               scripts that regenerate them from a run directory
```

## Run it

From this directory:

```bash
python3 -m http.server
```

Then open http://localhost:8000/. It must be served rather than opened as a `file://` URL, because
the page loads its data with `fetch` and uses ES modules.

## Controls

- **`▶ STORY`** is the one to start with — the five-act walkthrough, see below.
- Drag the timeline to scrub hour by hour (tick 1–336 = 14 simulated days).
- Play / pause at ½× / 1× / 3× / 8×. At 1× an hour takes 450 ms, which is slow enough to read the
  speech bubbles; 8× is a time-lapse.
- **`VS B2`** splits the screen into two offices on one shared clock.
- **`中文` / `EN`** switches the narration between Chinese and English, at any point including
  mid-act. Everything the run produced — issue titles, rule text, proposals, channel messages —
  stays in the original English, because it is evidence rather than copy. The choice is remembered
  in `localStorage`.
- Keyboard: `space` play/pause, `←` / `→` step one hour, `v` split screen, `s` story mode.

### What the office announces

There are no particle effects. Each room states in words what happened in it that hour, as a note
that rises off the room and fades:

- **MAINLINE — `merged`** — a PR landed on main.
- **SHIPPING — `shipped`** — `publish_product_release`; the rocket lifts at the same time.
- **PROTOCOL ROOM — `rule proposed` / `rule amended` / `vote`** — the governance actions.
- **CI LAB — `CI`** — any CI action. The machine also glows hotter the more CI runs.

Several of the same kind in one hour collapse into one note counted as `×n` rather than repeating.
Notes are drawn above the night dimming so they stay readable after dark, and they are the only
per-hour signal now, which is why they hold full opacity before fading.

The one remaining animation that means "something went wrong" is separate: in Act 3 the whole
cabinet shakes once when the organizational gate blocks a merge.

## Story mode — the run in five acts

`▶ STORY` (or `s`) plays the run as a narrative rather than a dashboard. Each act owns the screen:
the static acts hide the office, the live acts hide the side panels. `◀ back` / `next ▶` step
between acts, `exit` drops you back into free exploration.

**Acts never auto-advance.** Each one plays out and then stops on a prompt, with `next ▶` pulsing,
so the reader sets the pace between acts. Within an act the clock still moves on its own — that is
the animation.

1. **THE BRIEF** — the mission. All 12 cattrs issues in plain English, plus the reminder that the
   team is graded on 16 hidden tests it never sees.
2. **THE CAST** — the eight agents, with pixel portraits, their configured identity, top three
   skills, and their **declared failure modes**. Worth reading: `paul`'s include *bypass process*
   and *conflict with high-process agents*, `calvin`'s include *conflict with visionary push*. The
   fight in Act 3 is right there in the character sheets.
3. **THE FIRST RULE** — one complete governance cycle, played slowly, one hour at a time, with the
   governance panel spotlighting the card under discussion:

   > **t6** `calvin` proposes the first rule → **`victor` vetoes it**, nothing is created ·
   > **t12** `victor` proposes a narrower one · **t15** `paul` backs it, **adopted** ·
   > **t16** first time it binds · **t19** `calvin` violates it on `pr_14` and the
   > **organizational gate blocks the merge** · **t22** `victor` amends it ·
   > **t24** the system starts measuring whether it works.

   That is the whole loop: propose → veto → re-propose → adopt → bind → violate → enforce → amend.
4. **THE MACHINE** — the same loop at 45 hours/second, pausing only when another rule lands (t96)
   or fails to (t169, t271), while the blocked counter climbs to 648.
5. **THE LEDGER** — what shipped (10 of 12), the four surviving protocols with their rule text and
   block counts, and the 2-of-16 hidden-test result.

Acts 3 and 4 are derived from the data — the cycle is found by taking the protocol with the most
enforcements and reading its event timeline — so regenerating the JSON re-derives the narrative.
The narration itself lives in `i18n.js`, one entry per line of copy with an `en` and a `zh` side;
interpolated lines are functions of the values pulled from the run.

## Split screen (B2 vs B3)

`VS B2` puts the B2 office on the left and B3 on the right, both replaying the *same* seed on the
same clock. The two runs issue byte-identical actions until **t6**, then drift.

What to watch for:

- B2's protocol room is stamped **NO PROTOCOLS** and nobody walks into it — **0 visits in 336
  hours**, against 139 for B3, whose board fills with sticky notes as agents vote.
- The paired counters show B2 **ahead** on raw output for most of the run.
- The red mark on the timeline is t6, the first hour where the two arms diverge.

(`schedule_meeting` and `attend_meeting` are deliberately *not* mapped to the protocol room —
booking a meeting is not governance, and both arms do it. Leaving them in would have put B2 agents
in that room ten times and muddied the signal.)

## The workbench (default view)

The B3 view is three panels around the office, all driven by the same tick:

- **THE WORK** (left) — the 12 cattrs issues the team is graded on, in plain English
  (*"A whole number is rejected by a pass-through union that accepts floats"*). Cards move through
  open → in progress → built → merged, and the subtitle names the agent and action that moved it.
  Click a card for the full issue text and the complete handoff trail.
- **GOVERNANCE** (right) — **LIVE RULES** shows each protocol that exists by this hour with its
  live use / blocked counters; click for the rule text and the problem it was written against.
  **DECISIONS** is the proposal feed, where `ADOPTED` and `VETOED` land as they happen.
- **CHANNELS** (below the office) — the actual messages agents posted, verbatim.

Scrubbing backwards is exact: every panel is a pure function of the tick, not an accumulated log.

At t225, for instance, the three panels line up into one story: `sean` posts the union
pass-through fix to `#engineering`, the matching issue card flips to **merged** by `paul`, and that
issue is one of the two hidden contracts that ends up passing.

## What you are looking at

Eight agents work a real upstream migration task (cattrs v25.1.0 → behavior of v26.1.0) for 336
ticks, across five places work can happen — **code, review/merge, CI, protocol room, shipping** —
plus counters that tick up live as the run progresses.

The point of the run is the comparison in the footer: on the same seed, same model and same task,
the arms end at

| Arm | Hidden contracts passed |
|-----|-------------------------|
| B0 solo founder | 0 / 16 |
| B1 chat team | 0 / 16 |
| B2 roles only | 1 / 16 (`hidden_converter_alias_default`) |
| **B3 institutions** | **2 / 16** (`hidden_missing_handler_error_payload`, `hidden_union_passthrough_numeric`) |

The result is not "B3 does more work". On this seed B2 ends **slightly ahead on raw output** — 70
merged PRs and 47 releases against B3's 72 and 45 — with **zero** protocol machinery. B3 spends its
effort on **1 197 protocol uses and 648 enforcements**, and that is what buys it twice as many
hidden contracts. The split screen exists to make that trade legible.

## Data

`data/` holds two generated files, both committed, together about 0.6 MB. The demo needs nothing
else, so the directory is self-contained even without the raw runs.

Regenerating them requires the raw OrgEnv run directories, which are **not** in the repository.
Both scripts default to `scratch/log/<run name>` and take a flag if yours live elsewhere:

```bash
python3 build/build_timeline.py    # -> data/cattrs_b3_4013.json   (per-tick actions)
python3 build/build_b3_detail.py   # -> data/b3_detail_4013.json   (per-tick content)

python3 build/build_timeline.py --runs /path/to/runs
python3 build/build_b3_detail.py --run /path/to/runs/20260913-161353_org_mx4013_cattrs_b3_seed4013_llm
```

`build_timeline.py` reads `summary.txt` (per-tick actions), `figure_metrics.jsonl` (checkpoint
metrics) and the final evaluations, and emits per-tick cumulative counters so the numbers move every
hour instead of jumping every 24 ticks. The B2 run is parsed the same way into `compare_track`,
which is what the split screen replays. `replay.json.gz` (~400 MB) is deliberately not used.

`build_b3_detail.py` is what makes the workbench possible. It reads `final_snapshot.json` and
`organizational_capability_evidence.json` and pulls out 26 tasks with per-tick status history,
6 protocols with their rule text plus all 2 567 tick-stamped protocol events, 77 proposals with
their vetoes, and 235 messages.

Two things the data does **not** support, worth knowing before you read too much into the view:

- **There is no dialogue.** All 235 messages are broadcasts into 4 channels — zero DMs, zero
  `@`-mentions, zero replies. Agents coordinate through artifacts (PRs, reviews, protocol votes),
  not conversation. The feed shows this honestly rather than inventing threads.
- **Veto reasons are a stub.** All 37 rejections carry the identical string *"not convinced this is
  worth the cost"*. Who vetoed whom is real signal; the argument behind it is not in the data.

Note also that the two metric sources disagree slightly: `experiment_run_record.json` reports 1 140
protocol uses where `figure_metrics.jsonl` reports 1 197. The demo uses the checkpoint series
throughout so the live counters land exactly on the numbers shown in the footer.

## Caveat

This is a parity run on **DeepSeek V4 Flash**, not the paper's headline models, and one seed is not
the paper's aggregate. Use it to explain the mechanism, not to quote numbers.

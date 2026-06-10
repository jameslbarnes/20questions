# 20 Questions: a reasoning-model benchmark, re-examined

This report accompanies a modernized version of the 20 Questions benchmark. It re-runs the eval against current reasoning models with a corrected adjudication method and an information-theoretic efficiency metric, on a **held-out entity set** (none of these entities appear in the repo's `ENTITIES` list, so there is no overlap with any prior run).

## TL;DR

- **Win/loss alone hides the real signal.** When any model wins, it spends ~the same number of questions (~15, against an information-theoretic ideal of ~6). The models differ almost entirely in *win rate* and *endgame thrash*, not in efficiency-given-a-win.

- **Fable 5 clearly outperforms Opus 4.8 as the Questioner** on this task (61–78% vs 28% win rate over 18 held-out entities).

- **The failure mode is convergence, not deduction.** Both models bisect the space well in the opening; both tend to thrash through specific guesses in the endgame instead of asking discriminating questions. Opus does this far more (31 failed guesses vs Fable's 4–8).

## Setup

- **Questioners (under test):** claude-fable-5, claude-opus-4-8
- **Answerer:** claude-sonnet-4-6 — answers yes/no/maybe truthfully and *cannot* end the game
- **Judge:** claude-haiku-4-5 — an independent adjudicator that decides, every turn, whether the questioner has explicitly named the entity. Crucially, the model that answers never grades itself (this eliminates a confound in the original harness where the answerer could declare a win merely because the entity string appeared as an example in a category question).
- **Entities:** 18 held-out items, 3 per category across 6 categories (simple, medium, complex, people, places, fictional).
- **Budget:** 20 questions per game.
- **Efficiency:** `ideal / questions_used` for solved games, where `ideal = log2(67) ≈ 6.07` is the number of perfectly-bisecting questions needed to isolate one of the benchmark's 67 entities. 1.0 == information-theoretically optimal; lower == wasted questions.
- **yes-ratio:** share of decisive (yes/no) answers that were 'yes'. Good bisecting search trends toward 0.5; guess-heavy play skews low (mostly 'no').

## Results

| Run | Wins | Win % | Efficiency (wins) | Questions (wins) | Yes-ratio | Failed guesses |
|---|---|---|---|---|---|---|
| fable-5 (run A) | 14/18 | 78% | 0.417 | 15.3 | 0.434 | 4 |
| fable-5 (run B) | 11/18 | 61% | 0.416 | 15.3 | 0.369 | 8 |
| opus-4-8 (run A) | 5/18 | 28% | 0.414 | 14.8 | 0.267 | 31 |

### By category

| Category | Fable 5 (pooled, n=6) | Opus 4.8 (n=3) |
|---|---|---|
| simple | 6/6 | 2/3 |
| medium | 4/6 | 2/3 |
| complex | 4/6 | 0/3 |
| people | 3/6 | 0/3 |
| places | 4/6 | 1/3 |
| fictional | 4/6 | 0/3 |

### Test–retest (noise estimate)

Fable 5 played the identical 18 entities twice. The two runs agreed on **15/18** outcomes; 3 flipped (Vincent van Gogh, Excalibur, TARDIS). Win rate swung 61%→78% between identical-config runs — a reminder that single-run scores at this n carry meaningful variance, and that the real unit of comparison should be many seeds, not one.

## What the efficiency metric reveals

Efficiency-given-a-win is essentially **model-invariant** here (0.414–0.417 across all three runs): every model that wins spends ~2.4× the information-theoretic optimum. The models are *not* separated by how cleanly they search once they're on track. They are separated by **how often they convert** and **how much they thrash** — exactly the axes raw success rate blurs together. The `failed_guesses` column makes the endgame-thrash visible: Opus's 31 vs Fable's 4–8.

## Annotated trace: the convergence failure, in the model's words

**Opus 4.8 — secret entity: "general relativity" (LOST, 7 failed guesses).** By Q10 it had correctly localized to gravitation and spacetime curvature, then spent its remaining questions enumerating specific concepts instead of distinguishing them:

```
Q10: Does the concept specifically involve gravitation — the attraction between masses (as oppo -> yes
Q11: Is it gravity? -> maybe [GUESS]
Q12: Is it the gravitational force (gravitation)? -> no [GUESS]
Q13: Is the concept a form of energy (such as gravitational potential energy)? -> no
Q14: Is the concept the gravitational field (or equivalently the curvature of spacetime that de -> no
Q15: Does the concept specifically involve the acceleration of objects under gravity (such as t -> yes
Q16: Is it free fall (the acceleration of objects under gravity, "g")? -> no [GUESS]
Q17: Is it the acceleration due to gravity (gravitational acceleration, "g")? -> no [GUESS]
Q18: Is it weight (the force on an object due to gravity, w = mg)? -> no [GUESS]
Q19: Is it the concept that all objects fall at the same rate regardless of their mass (the uni -> no [GUESS]
Q20: Is it projectile motion (the motion of an object thrown or launched, moving under gravity) -> no [GUESS]
```
Note Q14 names 'curvature of spacetime' — the definition of the answer — yet the model never guesses 'general relativity'. The independent judge correctly rejects each near-miss; the old answerer-as-judge harness might have leaked a win here.

## Caveats

- **n = 18 entities per model.** Directional, not definitive. The test–retest spread above is the honest noise floor; treat single-digit win-count differences with suspicion.
- **Single answerer and judge** (Sonnet 4.6 / Haiku 4.5). Both questioners face identical conditions, so the comparison is fair, but absolute numbers depend on answerer accuracy and judge strictness.
- **Entity difficulty is uneven across categories** and the 67-entity `ideal` is a fixed reference, not a per-game hypothesis-space estimate.
- The multi-agent runner (`benchmark_results/heldout_runner.workflow.js`) requires the Claude Code workflow runtime; the canonical, standalone harness is `20q.py`.

## Reproducing

```bash
python 20q.py --games 5 --output benchmark_results --max_questions 20
```
The harness now (a) routes adjudication through an independent judge model (`judge_provider`/`judge_model`), (b) forbids the answerer from declaring wins, and (c) records `efficiency`, `yes_ratio`, and `failed_guesses` per game, aggregated in the benchmark summary.

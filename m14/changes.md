# Episode 14 — revision changelog (2026-08-29)

Result of a verify → critique → improve pass over `m14-fable-version.ipynb`, `m14-post.md` and `m14-promo.md`.

## Verification

- **Every metric reproduced bit-exact** across two clean re-executions (foundation-model inference is deterministic — quantile heads, no sampling — and the seeded Keras fits replayed identically): arena RMSE 11.55 / 9.46 / 9.37 / 8.30 / 8.76 / 7.74 / 8.01, coverage 0.79 / 0.79 / 0.80, pinball 1.98 / 2.10 / 1.86, all close-up scores, and the full cold-start table (5.62 … 8.47). Only wall-clock timings drifted, as expected (see below).
- Claim audit (all correct): parameter counts (1,190 linear = 84×14+14; 13,774 GRU), window arithmetic (16,370 = 1,637×10; 14,740/1,630 split; 19 cold-start windows), the 0.0977% attention-cost ratio (16²/512²), the 17%/33% Chronos-2 margins, per-library quantile column layouts (TimesFM mean+deciles → 1/5/9; TiRex deciles → 0/4/8; Chronos-2 requested levels → 0/1/2), the leak-free `context_for`/`actuals_for` boundary handling, and the group-mode per-series cost claim. The seasonal naive here already used the honest tiled construction (the m13 fix was the convention in this notebook from the start).

## Notebook changes

1. **Fixed a color-category error in the verdict bar chart** (the substantive catch of this pass). The palette was binary — green for zero-shot, blue for "everything else" — so **seasonal naive was painted the same blue as the trained specialists**, contradicting the chart's own title and the series convention (m12/m13 paint baselines red). Now three colors: green = zero-shot, blue = trained specialists, red = free baseline, with title and commentary updated ("every green bar is shorter than every blue bar, and the red one towers over both camps").
2. **De-duplicated the arena scoring machinery**: `arena_rmse` and `arena_prob` each rebuilt the identical 1,680-day ground-truth concatenation; it is now built once as `ARENA_ACTUALS` and shared, with the commentary cell updated to say so.
3. **Added the validation-overlap honesty note** to the specialist data-pipeline commentary: the "most recent 10% per item" split still shares 83 of 84 days across each boundary, so the validation loss is mildly optimistic — the same purged-gap caveat the m12 notebook carries, now consistent across episodes.
4. **Synced all hardcoded wall-clock numbers in markdown to the executed run**: linear 4.5→4.7s, GRU 23.5→23.7s, TimesFM 6.3→6.5s, TiRex 2.3→2.4s, Chronos-2 0.3→1.8s, group 0.8→1.0s. The Chronos-2 figure moved the most (its earlier 0.3s run had absorbed warm-up into the load call); it remains the fastest of the three, and the group mode remains cheaper per series (1.0s for 600 forecasts vs 1.8s for 120), which the text now states with the fresh numbers.

## Graphs

- All 12 figures regenerated from the re-executed notebook into `graphs/graph14-01.png` … `graph14-12.png` (same filenames and order) and spot-verified against prose claims (fanning uncertainty bands, the cold-start close-up, the recolored verdict chart).
- Visually changed: only `graph14-11.png` (the verdict bar chart — baseline now red, per fix #1). All other figures unchanged in content.
- Operational note: the first re-execution silently dropped the bar-chart display output (an nbclient capture glitch — the cell executed, outputs list came back empty); a second full run captured all 12 figures. Worth knowing if a future re-run comes up a figure short.

## Post changes (`m14-post.md`)

- **Notebook link fixed**: now points to `m14/m14-fable-version.ipynb` (same stale root-path pattern as the other episode posts).
- **Figure 10 caption and bar-chart sentence updated** for the three-color scheme (baseline in red).
- **Timing numbers synced to the executed run** in both specialist and foundation-model code blocks, the verdict table's `compute_s` column, and the TimesFM commentary ("six and a half seconds"). All RMSE/coverage/pinball numbers were already exact and stand unchanged.

## Promo changes (`m14-promo.md`)

- "In 0.3 seconds, on a laptop" → "In under two seconds, on a laptop" (the 0.3s was a warm-cache artifact); TimesFM "6.3s" → "6.5s". All other promo numbers verified against the run and unchanged.

## Editorial pass on the post (same day, second review)

A separate content review of `m14-post.md` focused on narrative flow, reader accessibility, and removing patterns that read as machine-generated. No numbers, figures, section structure, or jokes changed.

- **Narrative fixes** (the main work): the verdict → cold-start bridge claimed the arena was "a draw between paradigms" when the scoreboard had just shown a clear foundation-model win — rewritten to "a clear win, then — but a modest one … Is there a setting where the gap becomes a chasm? There is." The ladder framing now starts at the ladder: TimesFM is marked as "the first rung" (TiRex and Chronos-2 were already "the contrarian rung" / "the top rung"). "The ones we won't run" no longer opens with skip-this-section "For completeness…" — it now enters ("Three rungs climbed. Before the scoreboard, a quick look over the fence…") and exits ("Ideas noted — back to the three we actually ran.") with handoffs. The Closing-time recap chain broken into sentences, and the tokenization section's verbless fragment opener fixed.
- **Accessibility**: glossed inline — percentiles/quantiles (the load-bearing concept of the probabilistic story, previously undefined), "decoder-only" (contrasted with episode 13's encoder blocks, which is the only transformer variant the series had built), MLP, T5, PLC hardware, and the MAE/RMSE one-liner now standard across the m12–m14 posts. Episode callbacks (leakage liturgy, conformal calibration, transfer lesson) left as-is.
- **Style/detector tics** (lighter here than m12/m13): "honest" trimmed to brand-plus-calibration uses; "precisely/exactly" down to two literal uses; two of six "not X, but Y" template instances rewritten (kept the strong ones: "a finding, not a triumph", "don't degrade — they collapse"); the zoo paragraph's four-dash pileup reduced. Voice markers (";-)", ":-)", "good nose", "episode 7 says hello", "Read it slowly", the routing grid) deliberately preserved.

## Considered and deliberately not done

- Chasing the old 0.3s Chronos-2 timing by re-running until the warm-up lands in the load call: rejected — the notebook must quote the run it actually shows, and "fastest of the three" survives either way.
- Adding early stopping/validation to the cold-start specialists: their unregularized 200-epoch fits are the point of the demonstration (19 windows can't support training), and the text discloses it.
- `CFG.graph_folder` remains unused but is the series-wide CFG idiom — kept.

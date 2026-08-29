# Episode 13 — revision changelog (2026-08-29)

Result of a verify → critique → improve pass over `m13-fable-version.ipynb` and `m13-post.md`.

## Verification

- **One real protocol bug found** (see fix #1 below); everything else checked out. All seeded model results reproduced **bit-exact** on a clean re-execution (GRU 9.349, LSTM 9.328, vanilla 9.997, flat-head 9.365, linear 8.977, PatchTST 9.137, iTransformer 8.696; the full lookback-sweep table; the entire Melbourne rematch). Only wall-clock times wobbled, as expected.
- Arithmetic and claim audit (all correct): 56× parameter ratio of the flat head vs GRU, 196-fold attention-matrix reduction from patching, the 7% iTransformer-over-GRU margin and 0.28-RMSE margin over linear, 1,239→903 training-window counts across the sweep, 59,150 channel-folded samples, 1,190 and 203 linear-model parameter counts, the leakage-free `split_starts` logic, and the legitimacy of the lag-364 and persistence baselines.
- Attention-map extraction idioms verified against the model graphs (tokens re-embedded + PE re-added for the vanilla model; no PE for iTransformer — both match how the trained blocks actually received their inputs).

## Notebook changes

1. **Fixed a leaky lag-7 seasonal naive baseline** (error catch — the substantive fix of this pass). With a 14-day horizon, the original slice `z[s-7 : s+CFG.HORIZON-7]` used the *observed* days `s .. s+6` — the first forecast week's actuals, unknown at origin `s` — as the forecast for the second week. The honest baseline tiles the last observed week across the horizon (`np.tile(z[s-7:s], (2, 1))`). Corrected score: **MAE 8.662 / RMSE 11.301** (was 8.466 / 11.019, flattered by the leak). No ranking changes — lag-364 (10.601) already won and now wins by more. The explanation cell now teaches the trap explicitly: whenever the horizon exceeds the seasonal period, the tempting one-line slice quietly commits leakage.
2. **Fixed a factual error in the rematch commentary**: the GRU (3,591 params) was described as having "three orders of magnitude fewer weights" than the vanilla transformer (79,623) — the true ratio is ~22×, which the same cell later stated correctly. Now reads "a twenty-second of the weights" throughout.
3. **Eliminated the pandas/matplotlib date-converter `UserWarning`** that appeared in the GRU forecast-plot cell output: `plot_test_forecast` now draws the observed series with `plt.plot(obs.index, obs.values, ...)` instead of mixing `Series.plot` with `plt.plot`. Published notebook is warning-free; the figure is visually identical.
4. **Cleaned up unused label tensors**: `y_test_ci` and `yb_test` were computed but never consumed (scoring deliberately re-slices raw values); both are now `_` with a comment, so the code no longer implies those labels feed the evaluation.
5. **Disclosed the rematch head choice**: a sentence added to the contenders commentary noting the vanilla transformer enters the Melbourne rematch with its flat head — the stronger of its two arena-A variants, and cheap at 28 tokens.
6. **Sweep timing prose synced to the fresh run**: "~11s at four weeks, ~74s at a year" for the GRU, "~4–6s" for the iTransformer (was ~10/~77 and ~5–7).

## Graphs

- All 14 figures regenerated from the re-executed notebook into `graphs/graph13-01.png` … `graph13-14.png` (same filenames and order) and spot-verified against prose claims (bright-column channel-attention asymmetry, the GRU forecast overlay, the corrected lag-7 bar in the final chart).
- Visually changed: only `graph13-14.png` (final bar chart — the lag-7 baseline bar lengthened to its honest 11.30). All other figures are unchanged in content; `graph13-03.png` re-rendered identically from the warning-free plotting code.

## Post changes (`m13-post.md`)

- **Notebook link fixed**: now points to `m13/m13-fable-version.ipynb` (file lives in the `m13/` folder, not the repo root — same stale-link pattern as the other episode posts).
- **Baseline block updated** to the honest lag-7 numbers (8.662 / 11.301), with the surrounding paragraph rewritten: "beats a tiled copy of last week … twice", plus a new one-sentence teaching note about the tiling trap (the naive may not peek at week one's actuals to forecast week two).
- **"Three orders of magnitude" corrected** to "one-twentieth the weights" in the verdict of the Melbourne rematch (the intro's "one-twentieth its size" was already right).
- **Verdict table** updated: lag-7 row (8.662 / 11.301) and the `train_s` column synced to the fresh run.
- **Lookback-sweep cost sentence** synced (~11s / ~74s GRU, ~4–6s iTransformer).
- All other numbers verified unchanged and left as published. `m13-promo.md` checked — no affected numbers, no changes needed.

## Editorial pass on the post (same day, second review)

A separate content review of `m13-post.md` focused on narrative flow, reader accessibility, and removing patterns that read as machine-generated. No numbers, figures, section structure, or jokes changed.

- **Accessibility** (the main gap): **"token"** — the load-bearing word of the whole post, central to the intro's question and the core principle — was never defined; it now gets a one-line gloss where the question is first posed. Also glossed inline: MAE/RMSE (matching the m12 post), "embedding" (the vector standing in for a token), "add & norm" and attention "heads" (previously bare jargon in "four heads and two blocks"), persistence (in the Melbourne section), and "inductive bias". Episode callbacks (gates, leakage liturgy, error accumulation, purged folds) left as-is — they build on earlier posts by design.
- **Flow**: the attention section now opens by picking up the GRU-bottleneck argument the incumbents section planted ("the thing that promises to dismantle the GRU's bottleneck"); the Closing-time recap was broken from one semicolon mega-chain into readable sentences.
- **Style/detector tics**: thinned "honest/honesty" from six uses to two (intro brand line + takeaway 6); "exactly/precisely" from seven to four (kept only the literal/earned ones); varied three of the five "Two X." paragraph openers; de-duplicated "earn its keep" (kept the intro's) and "industrializes/industrialize" (twice in one closing paragraph); reduced em dashes so no paragraph carries more than three (the iTransformer inversion paragraph had five). Voice markers (";-)", "Publicly. By a lot.", "recalibrated everyone's dignity", "electricity bill", the away-game frame) deliberately preserved.

## Considered and deliberately not done

- Re-plotting the correlation matrix before the channel-attention map (cell 83) duplicates Figure 2, but keeping the two heatmaps adjacent in the notebook is a deliberate reading aid after ~60 intervening cells; the post already skips the duplicate. Left as is.
- The corrected lag-7 baseline could alternatively repeat lag-7 and lag-14 weeks (using `z[s-14:s]`); tiling the single most recent week is the standard `snaive`-style convention and simpler to teach. Chose tiling.
- `CFG.graph_folder` remains unused but is the series-wide CFG idiom — kept for consistency with m11–m15.

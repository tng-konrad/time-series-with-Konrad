# Episode 12 — revision changelog (2026-08-29)

Result of a verify → critique → improve pass over `m12-fable-version.ipynb` and `m12-post.md`.

## Verification

- **All results reproduced exactly.** The notebook was re-executed twice from a clean kernel (Keras 3.15 / torch backend / MPS); every metric matched the previously published numbers to the last decimal (e.g. seasonal naive 17.97, scratch 20.05, zero-shot 15.63, head-only 15.36, full FT 14.85, masked AE 20.15, contrastive 18.97, weather 20.78/20.66, and the full 60/120/240/480-day sample-efficiency table). The seed-before-build discipline holds, and the refactors below are numerically neutral.
- Arithmetic claims in the prose were checked and are correct: the 4.42-RMSE zero-shot-vs-scratch gap, ~156 gradient steps/epoch, ln(511) ≈ 6.2 for the NT-Xent chance level, window counts (86 = 69 + 17; 26 windows at 60 days), and the 12.0 / 0.4 RMSE gaps at 60/480 days quoted in the post and promo.
- The comparison table's row labels were audited against the code — the old positional bookkeeping happened to be correct, but see fix #2.

## Notebook changes

1. **De-duplicated the walk-forward test-set construction** (flow simplification). The same origin-loop was hand-rolled twice — once in Groundwork, once inside the sample-efficiency loop. It is now a single Utils helper, `make_test_windows(series, mean, std)`, with its own explanation cell; both call sites use it. This also makes the "each model must read the test period through the statistics it trained under" point structural: the statistics are explicit arguments. Utils intro updated ("Seven small helpers").
2. **Made the scoreboard bookkeeping robust** (latent-bug fix). The `uses source data` / `uses target labels` columns were positional lists that silently depended on the insertion order of the `results` dict — any reordering would mislabel every row. Replaced with explicit name-keyed dicts mapped onto the index, and the column renamed to `source pretraining` with clearer values (`none` / `supervised` / `self-supervised` / `mismatched`).
3. **Fixed the misleading final bar chart** (error catch). The old color rule (`'yes' in u`) painted the self-supervised methods the same green as supervised transfer, undercutting the chart's own punchline ("only green bars clear the red line") since two green bars sat right of the line. The chart now uses four colors — green (supervised, demand source), turquoise (self-supervised, demand source), blue (no pretraining), orange (mismatched weather) — with a proper patch legend. The punchline is now exactly true, and the closing markdown gained the sharper observation that the turquoise bars used the *same source* and still fell short: the pretext must match the task, not just the data.
4. **Added an honesty note on the train/validation split** (rigor). Stride-1 windows overlap the chronological boundary (the last train window and first validation window share 34 of 35 days), so the validation loss is mildly optimistic; the note explains why a purged gap is not affordable at 86 windows and why the untouched test period is the real verdict. Consistent with the episode-8 discipline the series teaches.
5. **Disclosed the fixed-augmentation simplification in the contrastive rung** (rigor). The two views are drawn once and reused for all eight epochs; full SimCLR re-samples per batch. One sentence added to the augmentations cell.
6. **Wording fix in the CFG cell**: `SOURCE_STRIDE` and `FT_LR` were jointly described as "learning-rate-adjacent knobs" — stride is not; now "two of the remaining knobs".
7. `from matplotlib.patches import Patch` added to the grouped imports cell (for the chart legend), with a matching clause in the imports walkthrough.

## Graphs

- All nine figures regenerated from the re-executed notebook into `graphs/graph12-01.png` … `graph12-09.png` (same filenames and order as before) and visually verified against the prose claims (near-flat AE reconstruction, fine-tune following the December decline, non-monotone scratch curve, etc.).
- Only `graph12-09.png` (the final comparison chart) is visibly different: four-color coding plus category legend, per fix #3. Figures 1–8 are unchanged in content.

## Post changes (`m12-post.md`)

- **Notebook link fixed**: now points to `m12/m12-fable-version.ipynb` (the file lives in the `m12/` folder, not the repo root). *Note: the m13/m14/m15 posts have the same stale root-path link — not touched here.*
- Figure references updated to the new cell numbers (the helper insertion shifted everything after cell 12 by two: figures now come from cells 26, 28, 40, 55, 58, 62, 67, 79, 87).
- Figure 9 caption rewritten to describe the four-color scheme.
- "The verdict" paragraph sharpened to match the new chart: "the only bars left of the red line are the green ones", plus the turquoise-bars observation (same source, wrong pretext).
- Rung 1: the train/val overlap honesty note added (mirrors notebook fix #4).
- Rung 6: the fixed-augmentations caveat added (mirrors notebook fix #5).
- **No numbers changed anywhere** — the re-run reproduced every metric exactly, so all quoted results, the scoreboard, the sample-efficiency table, and all Key Takeaways stand as published. `m12-promo.md` was checked and needed no changes.

## Editorial pass on the post (same day, second review)

A separate content review of `m12-post.md` focused on narrative flow, reader accessibility, and removing patterns that read as machine-generated. No numbers, figures, section structure, or jokes changed.

- **Flow**: the fine-tuning section now opens by answering the question rung 2 ends on ("They can."); a bridge paragraph marks the end of the ladder before the two zoom-out experiments, which are now introduced as "the data question" and "the source question"; the Closing-time recap was broken from one semicolon chain into readable sentences.
- **Accessibility**: added inline one-clause explanations of *window* (at first use, where a count was previously quoted before the concept), *MAE/RMSE*, *embedding space*, *instance discrimination* (previously used in a results paragraph without introduction), and *DTW distance*. Episode-link callbacks (purged gap, seq2seq skeleton, GRU gates) left as-is since they build on earlier posts by design.
- **Style/detector tics**: reduced the repeated "This is not X; it is Y" template from six instances to two (kept the strongest); thinned "honest/honesty" from eight uses to two; deduplicated "precisely/exactly" clusters; varied the four "Two X." paragraph openers and the twin "The picture…" figure leads; smoothed the fragment triad in the intro; cut em dashes in the densest paragraphs (the contrastive results paragraph went from five to one). Voice markers (emoticons, "weaponized", "borrowed brain", blunt verdict fragments) deliberately preserved.

## Considered and deliberately not done

- Purging the train/val window overlap: would cost ~⅓ of an already tiny training set and change every number; acknowledged in text instead.
- Re-sampling contrastive augmentations per epoch: needs a custom data pipeline, disproportionate for a 20-second pedagogical pretrain; disclosed in text instead.
- Removing `CFG.graph_folder` (unused): kept — it is the series-wide CFG idiom (m11/m13/m14/m15 all carry it).

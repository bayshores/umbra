# M7 held-out pipeline review, 2026-10-14

Reviewer: a fresh, read-only Claude Code agent (Opus 5.5). Code reviewed at git HEAD `de163f1`.
Files: `scripts/evaluation/m7/` (run_heldout.py, paint_heldout.py, common.py, evaluate_site.py, score.py) and `scripts/evaluation/metric_k.py`.
Extra care went on three commits: `4424fdb` (chunked S3 listing; box ground pass without suggestions), `c2f4f6b` (the D/E hook) and `de163f1` (metric A after the paint v4 review).

## Verdict: fix

"Fix" means at least one major problem was found. There are 2 major and 8 minor findings. Everything else I checked matched the lock text and the claims.

The details are in [verdict.json](verdict.json). Below is the plain-language summary.

## Major findings

1. **The labels addendum is not tied to the anchor or to a time.**
   - **What the lock says:** the addendum's sha256 must be committed in the tracked anchor and pushed *before any held-out file is opened*. Otherwise D and E stay "not scored".
   - **What the code does:** `preflight` checks only the protocol lock's sha256. The build receipt does not record whether an addendum existed. `score.py` and `evaluate_site.py` accept whatever file sits at the ignored path `lock/labels-addendum.json` when scoring runs.
   - **Why it matters:** an addendum written *after* held-out data was opened would be accepted. So would the D/E scorer it pins, which `evaluate_site.py` then runs. D and E would be scored.
   - **How I showed it:** I made a fake addendum and a fake "scorer" file inside this review folder, then pointed `score.LABELS_ADDENDUM` at the fake addendum (a monkeypatch, so the real path was never created). `score.py` scored D and E with it.
   - **Fix:** anchor the addendum sha256 (or an explicit "none") on origin/main and check it in preflight. Record it in each build receipt. Refuse D/E unless the build receipt, the anchor and the current file all agree.

2. **Fail-only evidence can be left out at one site, and nothing notices.**
   - **What happens:** `--paint` (metric A) and `--labels` (D and E) are optional for each site. Leaving one out gives "not scored".
   - **Why it matters:** A and E can only fail, never pass. So leaving out their input can only hide a fail.
   - **What the held-out summary checks:** that the site list is complete (the lost-site rule). It does not check that each held-out crossing has a paint record, or that each site in the committed label manifest has a D/E file or a failed D/E step.
   - **More gaps:**
     - The labels file is not checked against the committed label hashes.
     - The labels file's path is not recorded in the evaluate receipt.
     - The D/E reason ("the D/E step failed or no held-out labels") does not say which of the two happened.
   - **Why this needs fixing now:** score.py is frozen when the lock is approved, and the addendum may pin only the label protocol and the D/E code. A later addendum cannot add this check.
   - **Fix:** in held-out mode, the summary requires a paint record for every crossing. It also requires a D/E file, or a failed `labels-de` step, for every site in the committed label manifest. The manifest is checked the same way `lost_sites()` checks the lost-site list. The evaluate receipt records the labels file and its sha256.

## Minor findings

3. The `--out .local/../x` escape is fixed in paint_heldout.py but still works in run_heldout.py:366, evaluate_site.py:57 and score.py:562. I showed this with path arithmetic only; nothing was created.
4. run_heldout.py still catches only `SiteStop`. Any other error leaves no site record, and nothing blocks a rerun. One example is a listing that is not valid XML. evaluate_site.py also stops with no receipt if the pinned D/E scorer's sha256 is wrong.
5. Stop reasons in run_heldout.py and paint_heldout.py are not masked for the banned vehicle-status word, and `write_new` refuses any file that contains it. The frozen builder scripts each contain the word on a code line, so a traceback could quote it. The site would then end with no record. evaluate_site.py already masks for this reason.
6. metric E trusts the D/E file too much:
   - `appState` is not checked, so an `appState` of "Blocked" with a capital B hides a false Blocked.
   - E rows are not required to cover every F row.
   - Duplicate rows are read one way by E and another way by H.
7. metric D uses an empty list when `unresolvedLabels` is missing, and it does not check feature names or values.
8. The summary has no "each site passes D" field, although B has one. It also accepts site scores from different score.py versions or addenda.
9. H gives counts where the lock says rows are "listed by name". This is older than these commits.
10. There are no tests for the D/E hook or for the chunked-listing rule. The only M7 tests cover metric A.

## The 10 paint v4 review findings

| n | Status | Where |
|---|---|---|
| 1 | Fixed | A fails only when paint v4's nearEdge.status is "resolved" (score.py:137-139, lock text). |
| 2 | Honestly worded | There is no photo check, per Sean's Oct 14 choice ("count it against us"). The fail reason, the summary note and the lock text all say a look-alike line can cause a fail and that it counts against the app. |
| 3 | Fixed | The lock now defines the offset: field, sign, units and evalL. |
| 4 | Fixed | "Estimators disagree" and "not confirmed by paint v4" are now separate, and the summary lists them separately. |
| 5 | Fixed | score.py refuses a result.json not made by paint protocol v4 and code-v4. |
| 6 | Fixed in paint_heldout.py | The same defect remains in the other three drivers (minor finding 3). |
| 7 | Fixed in paint_heldout.py | The same defect remains in run_heldout.py (minor finding 4). |
| 8 | Fixed | Package versions are checked and recorded, and common.py's sha256 is recorded. |
| 9 | Fixed | The reason says "very likely at least about 0.20 m (not proven to be 0.30 m)". |
| 10 | Fixed | The summary states how many crossings were scored and that "not scored" says nothing about the map. |

## Claims I checked

- **The 8c rehearsal paint result equals paint v4's development result apart from configSha256 and the listing hash.** Confirmed: exactly those 2 differences.
  - The c780 result also differs in its `split` label ("development-optional" vs "development"). That label is copied from the config; it is not a measurement.
  - Both results carry the pinned paint v4 sha256 values.
- **The rescores have every outcome unchanged.** Confirmed: no outcome or reason changed.
  - 8c's H block gains only `eUsed` and the new rule text from c2f4f6b.
  - Both sites' A blocks gain `evalL` and `paintV4` from de163f1.
  - The rescore was made by HEAD's score.py.
- **The 8c rehearsal baseline equals the builder v4.3 development scene's in every metric value.** Supported, but not directly reproduced.
  - No stored development-scene baseline exists, and I was not allowed to rerun one.
  - Every input the baseline reads is the same: all binary buffers are byte-identical, and the JSON files differ only in paths and provenance hashes.
  - The rehearsal's B p95 (0.8975 m) matches the 0.897 m quoted in current.md.
- **The lock's metric A text matches score.py.** Confirmed: same field, limits, paint v4 condition, categories, refusal and caveats.
- **Metric D, metric E, H's reference rule and the falseBlocked bound match the lock text.** Confirmed by probe, apart from the gaps in minor findings 6 to 9.
  - The bound is 0.9747 for 1 false of 2 sites and 0.7764 for 0 of 2.
- **Nothing scores D or E without the addendum.** True as written, but any file at that path counts as the addendum (major finding 1).
- **4424fdb works as described.** Confirmed: only listings may arrive without a Content-Length, and the skipped suggester output is not needed by `car_shape.py plan`.

## Tests and probes

- `nice -n 10 .../python3 -B -m unittest discover -s tests/evaluation -p 'test_m7*.py'`: 6 tests, all OK. The git status was clean afterwards.
- `probes/jsondiff.py`: a deep JSON difference tool, used for the claims above.
- `probes/probe_de.py`: probes the D/E hook, the H override and the summary with a fake addendum. Output: `probes/probe-out/probe_de-output.json`. The fake files are in `probes/probe-out/`.
- `probes/row_states.py`: lists the app row states in the rehearsal F results ("blocked" and "unknown" only).
- `probes/probe_paths.py`: the `--out` escape check, using path arithmetic only.

## What I did not do, and process notes

- I did not open, list or download any held-out, control or reserve data, and I did not use the network.
- I did not read `.local/secrets` or any folder named by `PAO_LABELS_COORDINATOR_DIR`.
- I did not build scenes or rerun rehearsals or baselines. I edited no code and made no commits.
- I never created `lock/labels-addendum.json`. The probe stops if that file exists.
- **Slips, disclosed:**
  - While comparing the two scene folders, I wrote two temporary listing files to `/tmp` and deleted them right after.
  - Three Python runs used an inline `python3 -I -B -c` instead of a script in this folder. All three were niced and read-only: two read JSON fields, and one was an empty no-op.
  - Running the unit tests made the tests create and remove their own temporary folders, as the test file is written to do.

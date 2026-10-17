# Paint v4 for held-out metric A: independent review

Date: 2026-10-14. Reviewer: a fresh, independent, read-only reviewer (Claude Code, Opus 5.5) with no part in building paint v4, `paint_heldout.py` or `score.py`. I opened only development data, code and documents inside this repository. I did not open held-out, control or reserve data, and I used nothing from the network.

The machine-readable summary is `verdict.json` in this folder.

## 1. Verdict: fix

Paint v4 can serve metric A as a fail-only check, but **not as the claim, `score.py` and the lock text stand today**. Two changes are needed first. Neither one touches the frozen v4 code.

1. **`score.py` must respect v4's own decision (major finding 1).** Today metric A reads the two raw estimator numbers and ignores v4's own verdict on whether the edge was measured precisely enough ("resolved"). In v4's stored synthetic tests, this lets crossings whose paint is only 0.08 m and 0.17 m from the mapped edge be reported as failing. Counting a fail only when v4 says "resolved" removes those cases.
2. **The lock must deal with look-alike edges (major finding 2).** Some layouts fool both estimators in the same way, and v4 still says "resolved": a bright non-paint band (such as a concrete gutter) right in front of the crosswalk, a stop line almost touching it, or a faint, worn first stripe. In my synthetic probes these produced false fails at crossings whose true offset was within 0.10 m of the map. Fix 1 does not stop them. The smallest real fix is a blind, pre-registered look at the near edge in the camera images before scoring.

Everything else holds up. `paint_heldout.py` runs v4 unchanged, refuses held-out work without the approved lock, checks every download, never prints a measurement, and writes `paint-stopped.json` for every expected failure. It has a few small defects (section 10). The coordinator's rehearsal claim is correct (section 4).

`fitForMetricAFailOnly` in `verdict.json` is `false`, because the claim does not hold today. In my judgment it becomes true once fixes 1 and 2 (and the small wording fixes) are in place. Even then a "borderline zone" remains: a reported fail shows that the true offset is very likely at least about 0.20 m, not exactly 0.30 m.

If neither option in section 6 is adopted, my verdict would be **fail**.

## 2. Words used in this report

- **Crossing**: a marked pedestrian crosswalk at an intersection, as drawn in the Argoverse 2 (AV2) map.
- **Mapped near edge**: the line where the map says the crosswalk begins, on the side the car approaches from.
- **S**: distance in meters along the car's driving direction, measured from the mapped near edge. Negative means before the mapped edge; positive means past it.
- **Offset**: where the paint really begins, in S. 0 means the map is exactly right. +0.2 means the paint begins 0.2 m past the mapped line.
- **True offset**: the real answer. In synthetic (computer-made) tests it is known exactly; on real data it never is.
- **LiDAR, sweep, intensity**: the car's laser scanner. One sweep is one full turn of it. Each laser return has an intensity, meaning how much light came back. Paint is usually much brighter than asphalt.
- **Estimator A**: looks along narrow lengthwise strips for the first place where returns become much brighter than the road a few meters earlier. A return counts as paint when it is at least 3 times the strip's road median and also brighter than 90% of the same laser's road returns.
- **Estimator B**: looks, laser by laser, for the place where brightness steps up and stays up (a statistical test called Pettitt's test).
- **edgeSAtEval**: the number each estimator reports in `result.json`: its offset in meters, read at one sideways position called evalL.
- **evalL**: the sideways position where the offset is read. v4 uses the middle of the widest stretch that the scans covered well. That is not always the middle of the lane: at the 8c development crossing it is 0.52 m, while the middle of the lane corridor is 1.92 m.
- **v4 "resolved"**: v4's own final decision, written in `result.json` as `nearEdge.status`. It needs all of these: both estimators found an edge; when each of the 15 sweeps is left out in turn (15 "leave-one-out" runs), each estimator still finds an edge in at least 90% of those runs and its answer stays steady (a statistic called the jackknife standard deviation is at most 0.075 m) (together, the **precision rule**); and A and B agree within 0.15 m (the **agreement rule**). Only then does v4 publish its own offset (`offsetToMappedEdgeM`). "Marginally resolved" is a resolved edge where at least one leave-one-out run had A and B disagree, or where the average of A and B was not steady (over 0.075 m).
- **Metric A outcomes** in `score.py`: **fail** (both |A| and |B| at least 0.30 m, and within 0.15 m of each other); **unresolved** (A and B more than 0.15 m apart; counted, not scored); **not scored** (anything else). Metric A's "unresolved" is not the same thing as v4's "unresolved".
- **False fail**: metric A reports a fail although the true offset is well under 0.30 m. This is the harm that matters most.
- **Missed fail**: the true offset is 0.30 m or more, but metric A does not report a fail. For a fail-only check this costs power, not correctness.
- **R0, R1, R2**: the three versions of the metric A rule I compared. R0 = `score.py` as written. R1 = R0, plus v4 must say "resolved". R2 = R1, plus v4's label must be "resolved" rather than "marginally resolved".
- **Transplant check**: I took each synthetic trial's pair of errors (A's error and B's error) and added it to every true offset from -0.29 m to +0.29 m, to count how many would turn into fails. This assumes an estimator's error does not depend on the true offset. It is a rough "what if" check, not a measurement.
- **K**: in my probes, how many times brighter than the asphalt a non-paint band is.

## 3. The claim in plain words

At each held-out crossing, `scripts/evaluation/m7/paint_heldout.py` runs both frozen v4 estimators on the 15 sweeps that v4 itself picks. Metric A then reports a fail only when both estimators say the paint begins at least 0.30 m from the mapped edge and they are within 0.15 m of each other. The claim says this cannot produce a false fail, and that the tool runs v4 exactly as frozen, only under the approved lock, without printing or leaking measurements.

## 4. Rehearsal claim (check 5): confirmed

I compared the M7 rehearsal paint runs (`.local/evaluation/pao-accuracy/m7-rehearsal-2026-10-14/8c52d911-2658849-r3-paint` and `.../c780d53a-2069452-paint`) with the frozen development results (`.local/evaluation/pao-paint/results/<site>-v4.json`), field by field, with exact number comparison (`compare_rehearsal.py`, output `compare_rehearsal.json`).

| What | 8c52d911 / 2658849 | c780d53a / 2069452 |
|---|---|---|
| `result.json` fields that differ | `configSha256`, `inputs.sources.lidarListing` | `configSha256`, `inputs.sources.lidarListing`, `split` ("development" vs "development-optional") |
| Every measured value (A, B, v4 decision, offsets, leave-one-out results, coverage, diagnostics, sweep list) | identical | identical |
| `selection.json` | only `configSha256` differs | only `configSha256` differs |
| Config sources (poses, extrinsics, map, ground raster, ground transform) | same sha256 and same bytes | same sha256 and same bytes |
| LiDAR listing | different file, same 156 entries (key, size, ETag); the only extra element is `Delimiter` "/" | same 157 entries; same extra element |
| The 15 sweep files | match their receipts; same sha256 values as development | same |
| v4 step stdout and stderr files | empty | empty |
| Console log of the run | one line: site, "measured", record path, sha256 (no number) | same |

So the only differences are exactly the ones the coordinator named. Both runs used the current `paint_heldout.py` (sha256 `32fd6e94...`) and Python 3.13.5; every sweep downloaded on the first attempt with MD5 equal to ETag.

One small note. The 8c paint run finished at 12:21 UTC, about five minutes before the commit that changed the download code in `common.py` (4424fdb, 12:26 UTC). The paint receipt records the driver's sha256 but not `common.py`'s, so I cannot prove which `common.py` did the 8c downloads. The results are identical anyway, and held-out runs are protected because the lock pins `common.py` (minor finding 8).

## 5. Major finding 1: `score.py` ignores paint v4's own "resolved" decision

**Files**: `scripts/evaluation/m7/score.py` (`metric_a`, lines 103 to 126) and the metric A text in `.local/evaluation/pao-accuracy/lock/write_protocol_lock.py` (around lines 264 to 271).

### What happens

v4 was built to publish an offset only when the edge passes both its agreement rule and its precision rule. Otherwise `result.json` leaves `offsetToMappedEdgeM` empty and sets `nearEdge.status` to "unresolved". `score.py` never looks at that decision. It takes the two raw estimator numbers (`estimatorA.edgeSAtEval`, `estimatorB.edgeSAtEval`) and applies only the 0.15 m agreement rule. So a crossing where the two estimators happen to agree, but jump around when single sweeps are left out, can still be reported as a fail.

v4's own protocol (`scope.heldOutUse` in `protocol-v4.json`) says held-out use must reuse "the same sweep count, thresholds, agreement and precision rules". Metric A as written keeps the agreement rule and drops the precision rule.

### Evidence

Recomputed from the 1,080 stored rows of `.local/evaluation/pao-paint/validation/synthetic-validation-v4.json` (6 crosswalk patterns x 9 conditions x 20 trials; 320 trials have a true offset under 0.30 m). Scripts: `probe_synthetic_metric_a.py`, `probe_rule_variants.py`, `probe_joint_status.py`.

| | R0 (`score.py` today) | R1 (also needs v4 "resolved") |
|---|---|---|
| Fails | 519 | 275 |
| False fails (true offset under 0.30 m) | 3: true 0.084 m (ladder crosswalk, c780-like brightness; A 0.359, B 0.405); true 0.171 m (ladder, low contrast; A 0.576, B 0.501); true 0.292 m (bars, worn paint; A 0.316, B 0.321) | 1: the 0.292 m case only |
| v4's decision on those false fails | 0.084 and 0.171: unresolved (both estimators failed v4's precision rule); 0.292: resolved | resolved |
| Fails that v4 itself calls unresolved | 244 of 519 | 0 |
| Transplant: most fails at any true offset within +/-0.20 m | 15 of 1,080 (at +0.20 m); 4 at a true offset of exactly 0 | 0 (the first fails appear at +0.21 m: 2, and at -0.25 m: 3) |

There were 15 trials in which both estimators were wrong in the same direction by 0.10 m or more while agreeing within 0.15 m: 9 ladder crosswalks, plus 6 other crosswalks with worn paint (5) or low contrast (1). v4 called all 15 "unresolved". This is the kind of trial that R0 can turn into a fail and R1 always blocks.

The development crossings fit this picture. With only half the sweeps, 8c's estimator A jumped to 1.914 m (B found nothing), and c780's A moved to 0.353 m (B 0.137 m). Fewer good sweeps make A unstable, and v4's precision rule is the guard against that.

### Fix (small; v4 stays frozen)

In `score.py` `metric_a`, before returning a fail:

```python
    if abs(a) >= A_OFFSET_M and abs(b) >= A_OFFSET_M:
        near = doc.get('nearEdge') or {}
        if near.get('status') != 'resolved':
            return not_scored(f"paint v4 did not resolve the edge ({near.get('reason')}); possible fail not confirmed",
                              v4Unresolved=True, **offsets)
        return outcome('fail', ...)
```

In the summary, list those crossings by name, for example `'A': {..., 'notConfirmedByPaintV4Sites': [...]}`.

In the lock's metric A text, add: "A fails at a crossing only when paint v4 resolves the near edge (`result.json` `nearEdge.status` is 'resolved', which applies v4's agreement and precision rules), both offsets are at least 0.30 m, and they agree within 0.15 m." Also define "offset" (minor finding 3). R2 (label must be "resolved") is a slightly stricter option with almost the same numbers (274 fails instead of 275, the same single borderline false fail).

### What the fix costs

Fewer real fails are caught. Among the 623 stored trials with a true offset of 0.45 m or more:

| Crosswalk pattern | R0 catches | R1 catches | Out of |
|---|---|---|---|
| Continuous | 86 | 51 | 103 |
| Continuous, slightly skewed | 79 | 56 | 89 |
| Bars 0.3 m on 0.75 m | 85 | 41 | 105 |
| Bars 0.6 m on 1.2 m | 75 | 46 | 96 |
| Ladder (border band, then bars) | 77 | 25 | 115 |
| Stop line (0.3 m wide, ending 1.2 m before a continuous crosswalk) | 27 | 0 | 115 |
| All | 429 (69%) | 219 (35%) | 623 |

For a fail-only check this is acceptable: a missed fail becomes "not scored", never a wrong "pass". Reports must say plainly that "not scored" is not a pass.

## 6. Major finding 2: look-alike edges fool both estimators together, and v4 still says "resolved"

**Files**: the metric A text in `write_protocol_lock.py`; `score.py` too, if the recommended fix is adopted.

### What happens

Both estimators read the same laser returns. v4's own protocol says so: A-B agreement "is not independent confirmation". If something other than the crosswalk's first stripe is the first bright thing in the right place, both estimators can mark it together. They then agree with each other, they pass the precision rule (the error is steady from sweep to sweep, not noisy), and v4 often still says "resolved". Fix 1 does not help in those cases.

v4's stored validation has no such layouts. I built them with a copy of v4's own test generator and ran the frozen code with the frozen parameters (`probe_material_band.py`, `probe_faint_near_edge.py`, `probe_limit_line_gap.py`). True offsets were drawn between -0.10 m and +0.10 m, so every fail below is a false fail.

### Evidence

"Early" means the estimator reported the edge before the true edge; "late" means after it.

| Layout before or at the crosswalk | Road | Trials | Fails, R0 | Fails, R1 (v4 resolved) | What the estimators do in the fail trials |
|---|---|---|---|---|---|
| Bright non-paint band, 0.6 or 1.0 m wide, ending 0.1 m before the crosswalk, K = 2 to 3 | clean and dark | 48 | 2 | 0 | K = 2: the band is mostly ignored. K = 3: A usually jumps to the band but B often does not, so they mostly disagree. v4 called both fails unresolved |
| Same band, K = 4 to 5 | clean | 24 | 24 | 20 | Both jump to the start of the band: 0.6 to 0.73 m early (0.6 m band) or 1.0 to 1.2 m early (1.0 m band) |
| Same band, K = 4 to 5 | dark, 8c-like | 24 | 23 | 14 | Same |
| First 0.3 m of the crosswalk faint (0.4 or 0.25 times the paint brightness) | clean | 12 | 6 | 5 | Both start at the bright paint: 0.25 to 0.52 m late (values +0.30 to +0.45 m) |
| Same | dark, 8c-like | 12 | 3 | 0 | Same; v4 called all 3 unresolved |
| 0.3 m stop line, 0.1 m before the crosswalk | clean and dark | 12 | 8 | 1 | Both start at the stop line: 0.31 to 0.45 m early (values -0.31 to -0.51 m) |
| 0.3 m stop line, 0.3 m before | clean and dark | 12 | 1 | 0 | v4 called the fail unresolved |
| 0.3 m stop line, 0.6 m before | clean and dark | 12 | 0 | 0 | |

Example (clean road, 0.6 m band 5 times as bright as the asphalt): true offset -0.0003 m, so the map is right; A -0.704 m, B -0.706 m; v4 "resolved"; metric A "fail". The dark 8c-like road gave the same picture (A -0.700 m, B -0.702 m, resolved, fail).

The faint-stripe case depends on what counts as "the edge". If a worn but visible first stripe counts as paint (a person looking at the street would say so), these are false fails. If only bright paint counts, they are not.

### How bright are real surfaces?

I could not measure a real gutter in front of a crosswalk: that would mean opening data I am not allowed to open. The best scale I have is the earlier review's probe of the 8c development crossing (`.local/evaluation/pao-paint/review-v4/tail_probe-8c.json`, `roadLateralProfile`). It covers the road reference window, 3 to 7 m before the mapped edge, in 0.5 m wide sideways bins:

- lane asphalt (sideways positions -1.5 to 3.5 m): median intensity 2 to 3, 90th percentile 3 to 5.5. One bin (2.0 to 2.5 m) has a 90th percentile of 21, probably a lengthwise lane line;
- the last 1.5 m toward the curb (3.5 to 5.5 m): medians 5 to 11, 90th percentiles 9 to 13. The earlier review's probe names "the gutter next to the curb" as an example of such bright surfaces; I did not confirm what this surface is;
- paint at 8c: about 13 (the level used in v4's dark-road test model).

So a real non-paint surface at 8c is 2.5 to 5.5 times as bright as the lane asphalt. That covers both the harmless range in my probes (K = 2 to 3) and the harmful range (K = 4 to 5). That strip runs along the curb, not across the lane, so it only shows what brightness real surfaces reach in this data. Concrete cross gutters across the lane exist at some Bay Area intersections; that is general knowledge, not something I checked for Palo Alto here.

### Why v4's checks do not catch it

The agreement rule cannot catch it, because both estimators read the same returns. The precision rule cannot catch it, because leaving one sweep out does not remove the band: the error is the same in every sweep. Nothing in v4 asks whether the first bright thing is really the crosswalk.

### Fix options (v4 stays frozen)

- **Recommended: a blind near-edge surface check, written into the lock.** For each held-out crossing, before anyone reads its paint result, a person or a fresh agent who has not seen any paint output looks at the camera images the site already downloaded and answers one question: "Is the crosswalk's near edge, on the approach side, a clean change from asphalt to the first painted stripe?" Answers: **clean**; **not clean** (a light concrete band, patch or plate within about 1 m before the first stripe; a stop line within about 0.6 m of it; a worn, faint or old "ghost" first stripe; a raised crossing ramp with markings; a bike box); or **cannot tell**. The answers are committed under `scripts/evaluation/m7/anchors/` before scoring, the same way the held-out label hashes are. `score.py` then allows a fail only where the answer is "clean"; anywhere else metric A is "not scored" with that reason. This lowers coverage further, but it removes the false-fail paths I found.
- **Fallback: change what a fail claims.** Re-word metric A as: "A fail means the LiDAR brightness onset that both paint v4 estimators find is at least 0.30 m from the mapped edge. A bright non-paint surface, a stop line touching the crosswalk, or a faint first stripe can move that onset without any map error, so a fail is evidence to check, not proof that the map is wrong." This keeps the report honest, but it does not remove the false fails as the claim defines them.

The 0.6 m and 1 m distances come from my probes: a stop line 0.6 m away never caused a fail, while at 0.1 m it often did (at 0.3 m once, and v4 called that case unresolved). I tested bands only at 0.1 m from the crosswalk, so "about 1 m" is a cautious margin, not a measured limit.

## 7. Failure modes (check 1)

"Today" means R0 (`score.py` as written). "With fix 1" means R1.

| Failure mode | Today | With fix 1 | Evidence |
|---|---|---|---|
| Wrong mapped-edge frame or sign | Sign error: impossible as a false fail. Wrong edge: missed fail | Same | Metric A uses absolute values, so flipping the sign of both numbers changes nothing. Opposite signs cannot fail: if both are at least 0.30 m in size, they are at least 0.60 m apart, so metric A says "unresolved". v4 picks the near edge by a fixed rule (the first crossing edge met along the approach heading; heading within 30 degrees of the edge normal). If it picked the far edge, the paint would end there instead of begin, giving "onset at window start" or no edge, so not scored. The development frames look right: the mapped far edge is at +3.3 to +3.8 m. At c780 both the near edge (+0.18 m) and the far edge (+0.17 to +0.20 m) of the paint lie inside the mapped polygon, which is what a slightly wide polygon looks like; a shifted frame would move one end in and the other out. |
| Stop line at the usual distance (0.6 m or more before) | Missed fail | Missed fail | Stored validation (a 0.3 m stop line ending 1.2 m before a continuous crosswalk): 0 false fails; R1 catches 0 of 115 large offsets. My 80 trials with bar crosswalks plus that stop line, or with a 0.6 m wide stop line (also ending 1.2 m before), at true offsets between -0.25 and +0.25 m: 0 fails. A locks onto the stop line (median error -1.45 to -1.80 m) while B stays on the crosswalk (median error -0.00 to +0.06 m), so they disagree. |
| Stop line almost touching the crosswalk (0.1 m) | False fail (8 of 12) | False fail, rarer (1 of 12) | `probe_limit_line_gap.py`; major finding 2. |
| Lane lines (lengthwise) | Missed fail, or a little late | Same | They, and other bright lengthwise surfaces such as the strip toward the curb, raise A's per-laser road reference. The earlier review measured the effect at 8c (`review-v4/tail_probe-8c.json`): A is 0.101 m as frozen, and 0.029 m when each laser's "brightest 10%" threshold is taken only from road returns inside the well-covered stretch, so the frozen A is about 0.07 m late. Late pushes toward a fail only when the true offset is already positive and close to 0.30 m (minor finding 9). |
| Second line of a two-line crosswalk | Missed fail (never scored) | Same | 16 trials (`probe_two_line.py`): B never reported an edge (in each trial it found a step in at most 1 of its 44 strips, too few to report), so metric A is never scored. A alone reported values far past the first line in 5 of 16 trials (+1.40 to +2.74 m; the second line starts 2.7 m after the first). |
| Ladder crosswalk (border band, then bars) | False fail (2 of the 3 stored false fails) | Closed in the stored evidence | Both were v4-unresolved. |
| Bars | Borderline only (true 0.292 m) | Same | Stored validation. |
| Poor sweep coverage | False fail possible (unstable estimates that agree by chance) | Mostly missed fail | v4 needs 15 sweeps (fewer candidates means stopped, not scored) and the precision rule. With half the sweeps: 8c A 1.914 m, B none; c780 A 0.353 m, B 0.137 m. |
| Car in the corridor | Missed fail | Missed fail | Reasoned from code, not tested. Only returns within 0.10 m of the ground model are used, so a car body is never "paint". A gap of more than 0.5 m before an onset makes that strip "not localized" for A. Less coverage leads to "unresolved". |
| Curb | Missed fail | Missed fail | The corridor stops 0.25 m inside the drivable-area boundary, and the ground band drops the curb face. A bright gutter beside the curb raises A's road reference in those strips and makes A a little late (as for lane lines). |
| Very dark asphalt | Missed fail (not scored) | Same | Stored dark 8c-like, E1-8c and E2-8c conditions and my probes: no false fail caused by dark asphalt. |
| Very bright asphalt or low contrast | False fail possible together with a ladder crosswalk (the stored 0.171 m case) | Missed fail | Low contrast mostly gives "no edge" or "unresolved". |
| Bright non-paint band in front of the crosswalk (not in the task list; found here) | False fail | False fail, not closed | Major finding 2. |
| Faint first stripe (not in the task list; found here) | False fail | False fail, not closed | Major finding 2. |
| Worn paint in general | Both estimators about 0.04 to 0.06 m late, even when resolved | Same | Stored worn-paint trials that v4 resolved: continuous crosswalk, mean error A +0.053 m and B +0.057 m (9 trials; v4's own resolved offset is +0.055 m late on average, at most 0.10 m); slightly skewed crosswalk, A +0.035 m and B +0.039 m (9 trials). Part of the borderline zone (minor finding 9). |

**Development crossings** (`probe_dev_results.py`): both are v4 "resolved" and both are "not scored" under R0 and R1. 8c: A 0.101 m, B -0.009 m; it would need 0.29 m more to fail (0.26 m in the closest leave-one-out run). c780: A 0.194 m, B 0.166 m; 0.13 m more to fail (0.11 m in the closest leave-one-out run). No half-run (even, odd, near or far sweeps) fails either.

## 8. `paint_heldout.py` (check 2)

File: `scripts/evaluation/m7/paint_heldout.py`, sha256 `32fd6e94...`, unchanged since commit 2e11631.

| Requirement | Result | Evidence |
|---|---|---|
| Runs code-v4 byte for byte | Pass | Checks the sha256 of code-v4 (`bd2c6996...`) and of protocol-v4 (`f9a52b44...`) before running anything (line 82); a mismatch stops the run with `paint-stopped.json`. v4 imports only the Python standard library, numpy, scipy and pyarrow, so the one file hash covers all of its code. v4 also writes its own sha256 into `result.json`. For held-out runs, preflight checks the same pins from the lock. |
| No parameter overrides | Pass | Passes only `--protocol`, `--protocol-sha256`, `--config`, `--sweeps` and `--output`. v4 has no override options and reads only `schemaVersion`, `sources`, `logId`, `crossingId` and `split` from `config.json` (`split` is recorded, never used). |
| v4 selection with count 15 | Pass | Checks that the protocol's count is 15 (line 85) and that 15 sweeps were chosen (line 121). v4 picks the sweeps itself; its measure step re-runs the selection and checks each sweep's sha256 against `receipts.json`. Too few candidates means stopped, not scored. |
| Download size limit | Pass | Refuses a listed size over 4,000,000 bytes before downloading (line 128); curl `--max-filesize`; byte count == Content-Length == listing size. |
| MD5 and ETag | Pass | HTTP ETag == listing ETag; MD5 of the bytes == ETag. |
| Single-part objects only | Pass | An ETag containing "-" (a multi-part upload) is refused before downloading (line 128). |
| Retries | Pass | At most 3 retries, each into a new empty `attempt-<n>` folder; after that the site stops at step "download". curl uses `--fail`, `--proto =https`, `--max-time 300`. |
| Refuses held-out runs without the approved lock | Pass | The site folder's mode and lock must match the arguments (lines 61 to 64). Then the guard runs before any download or folder creation: lock file at its fixed path with the given sha256; an approval (by, utc); clean git status; `pipeline.commit` is an ancestor of HEAD with only anchor files changed since; `git fetch origin main` and the anchor there names this lock; every pinned file matches (paint code-v4, protocol-v4 and the M7 code included). Reserve sites are always refused; rehearsal runs only on development sites. |
| Writes `paint-stopped.json` instead of failing the build | Pass for every expected failure | Every expected failure raises `SiteStop`, which is caught and written as `paint-stopped.json` (exit code 3). The scene build is a separate program and is never affected. Unexpected errors are not caught (minor finding 7). |
| Never prints a measurement | Pass | Prints one line: site, outcome ("measured" or "not-scored"), record path and sha256. v4 has no print statements; each step's stdout and stderr go to read-only files; the rehearsal console logs and step files contain no number. |
| Writes only in its new output folder | Pass, with one gap | Refuses an existing `--out`; every file is created new (O_EXCL) and read-only. Gap: the "must be under `.local/`" test can be escaped with ".." (minor finding 6). |

**Differences from the development runs.** None of them changes a measured value; the rehearsal shows that.

- `config.json` is rebuilt from the site's receipted downloads, so its paths, provenance text and sha256 differ.
- The LiDAR listing is fetched with a "/" delimiter (one extra XML element, the same entries).
- `split` comes from `split.json` (c780: "development-optional").
- Sweeps are downloaded by `common.fetch` with listing size, ETag and MD5 checks (the development runs used `acquire.py`).
- v4 runs as a child process with `nice -n 10` and `python -I -B` from the repository root.
- The receipt records the driver's sha256 and the Python version, but not `common.py`'s sha256 or the numpy, scipy and pyarrow versions (minor finding 8). Today they match paint v4's lock: Python 3.13.5, numpy 2.3.1, scipy 1.16.0, pyarrow 20.0.0.

Not a defect: `common.LOG_PREFIX` only allows AV2's `train` split, so a held-out log stored elsewhere would stop, not be measured wrongly.

## 9. Metric A text and `score.py` against v4's output (check 3)

| Item | Lock text | `score.py` | v4 `result.json` | Consistent? |
|---|---|---|---|---|
| Which number | "offset", never defined | `estimatorA.edgeSAtEval` and `estimatorB.edgeSAtEval` | The same fields exist. v4's own held-out number is `offsetToMappedEdgeM`, left empty unless resolved | The field is right, but the text does not name it and the code skips v4's gate (major 1, minor 3) |
| Units | meters | meters (0.30, 0.15) | meters | Yes |
| Sign | absolute values | `abs()` | positive = paint starts past the mapped edge | Yes; the sign cannot cause a fail |
| Opposite signs | not mentioned | Cannot fail: they differ by at least 0.60 m, which gives "unresolved" | | Yes |
| Disagreement over 0.15 m | "unresolved (counted, not scored)" | not scored, flagged `unresolved` | v4 uses "unresolved" for any failed gate | Same word, two meanings (minor 4) |
| Precision rule (spread at most 0.075 m; an edge in at least 90% of leave-one-out runs) | not mentioned | not applied | `nearEdge.status` | No (major 1) |
| No edge | not scored | not scored | `edgeSAtEval` missing | Yes |
| The 1e-12 tolerance | not mentioned | agreement uses 0.15 + 1e-12, the same boundary as v4; 0.30 has no tolerance | v4 uses 0.15 + 1e-12 and 0.075 + 1e-12 | Yes; ">= 0.30" matches the text |
| Where the offset is read | not mentioned | at evalL, as v4 reports it | evalL = middle of the longest well-covered stretch (8c: 0.52 m; corridor middle 1.92 m) | Not stated (minor 3) |
| NaN (not a number) | | | v4 writes JSON with `allow_nan=False`, so it cannot occur | Yes |

## 10. Minor findings

Findings 1 and 2 are the majors above.

3. **"Offset" is not defined** (`write_protocol_lock.py`, metric A text). Fix: "offset = each estimator's `edgeSAtEval` in `result.json`: meters along the travel direction from the mapped near edge to the paint onset, positive when the paint starts past the mapped edge, read at `result.json` `frame.evalL` (the middle of the longest well-covered stretch, not necessarily the middle of the lane)."
4. **Two meanings of "unresolved"** (`score.py` summary and the lock text). Metric A's "unresolved" means "the estimators disagree by more than 0.15 m"; v4's means "any of v4's gates failed". Fix: call metric A's category "estimators disagree", and list v4-unresolved would-be fails separately (fix 1 does this).
5. **`score.py` does not check what actually ran** (`score.py`, `metric_a`). It checks `result.json` against the paint receipt, but not `result.json`'s own `protocolSha256` and `scriptSha256` against the pins. Fix: two equality checks (`f9a52b44...`, `bd2c6996...`), refusing otherwise.
6. **`--out` can escape `.local/`** (`paint_heldout.py`, lines 56 to 57). `Path(...).absolute()` keeps "..", and `is_relative_to` only compares text, so `.local/../scripts/x-paint` passes the test and the run would create `scripts/x-paint`. In a held-out run this could also make the next preflight fail, because git status would no longer be clean. Fix: use `.resolve()` on both paths, or refuse any ".." part.
7. **Unexpected errors leave no record** (`paint_heldout.py`, lines 139 to 141). Only `SiteStop` is caught. An unexpected error (for example a listing that is not valid XML, a full disk, or a missing code file in rehearsal) ends the run with neither `paint-receipt.json` nor `paint-stopped.json`. If `score.py` is then given that folder, it stops the whole site's scoring ("has no single paint record"), and nothing in the code prevents a second run into a new folder, although the `SiteStop` rule says a site is never rerun. The inputs are fixed and hash-checked, so a rerun would give the same numbers; the problem is procedure, not values. Fix: after the folder is created, catch any exception and write `paint-stopped.json` with step "unexpected" and the error type.
8. **Package versions and `common.py` are not recorded** (`paint_heldout.py`, lines 142 to 151). The receipt has the driver's sha256 and the Python version only. Fix: record and check the numpy, scipy and pyarrow versions against paint v4's lock environment, and record `common.py`'s sha256.
9. **Borderline zone** (lock text and report wording). Even with fix 1, a fail shows that the true offset is very likely at least about 0.20 m (transplant: the first fails appear at +0.21 m), not exactly 0.30 m. One resolved synthetic fail had a true offset of 0.292 m, and worn paint makes both estimators about 0.04 to 0.06 m late. Fix: say so next to any reported fail.
10. **Low power, and "not scored" is not a pass** (lock text and summary). Two-line crosswalks are never scored (B never reports an edge), crosswalks with a stop line at the usual distance never fail under fix 1, and R1 catches about 35% of synthetic offsets of 0.45 m or more. Fix: report how many crossings were scored, and say plainly that "not scored" says nothing about the map.

## 11. What I did not test

- Real gutters, patches or plates in front of a crosswalk. My bands are synthetic; real brightness comes only from the 8c curb strip.
- Bands at distances other than 0.1 m before the crosswalk.
- Uneven wear, for example paint worn mainly in the wheel tracks.
- Old "ghost" markings near the edge.
- Raised crossings and their ramp markings.
- Bike boxes and green paint. I expect "onset at window start" or no edge (not scored), but I did not test it.
- A vehicle standing or moving in front of the crosswalk (reasoned from the code only).
- Wet roads.

## 12. Tests I ran

All synthetic probes load the frozen code-v4 and the protocol-v4 parameters after a sha256 check, and use a copy of v4's own test generator (`code-v4/test_measure_paint_edge.py`). Each ran alone, small, with `nice -n 10` and Python `-I -B`.

1. `probe_synthetic_metric_a.py`: metric A rule R0 on all 1,080 stored synthetic-validation-v4 rows: 519 fails, 3 false fails.
2. `probe_rule_variants.py`: R0, R1 and R2; false fails; power per pattern; transplant check.
3. `probe_joint_status.py`: trials where both estimators were wrong the same way while agreeing, and v4's status for them (all 15 v4-unresolved).
4. `probe_dev_results.py`: the stored development results: values, margins to a fail, leave-one-out ranges, halves, far-edge diagnostic.
5. `compare_rehearsal.py`: rehearsal against development, field by field. Its `sweepShaComparison` entries are empty (`null`) because my helper looked for the sweep list under the wrong key. I compared the 15 sweep sha256 values in `result.json` with the download receipts directly instead (15 of 15 equal for each site), and the full `result.json` comparison, which walks every field including the sweep list, finds no other difference. I also read both paint receipts: driver sha256 `32fd6e94...`, Python 3.13.5, all 15 downloads on the first attempt with no earlier errors, bytes equal to Content-Length, and "single-part: MD5 == ETag".
6. `probe_untested_patterns.py`: 80 trials, bar crosswalks with a stop line and a 0.6 m wide stop line: 0 fails.
7. `probe_two_line.py`: 16 trials, two-line crosswalk: never scored.
8. `probe_material_band.py`: 48 trials at K = 2 to 3, then 24 clean and 24 dark trials at K = 4 to 5.
9. `probe_faint_near_edge.py`: 24 trials.
10. `probe_limit_line_gap.py`: 36 trials.
11. `env_versions.py`: Python and package versions against paint v4's lock.
12. `check_readme_numbers.py`: re-reads this folder's stored probe outputs (no new measurement) to check the row-level numbers quoted in sections 6 and 7.
13. Code reading: `paint_heldout.py`; `common.py` (guard, preflight, fetch, read_listing); `score.py` (`metric_a`, summary); `evaluate_site.py` (how the paint folder is passed); code-v4 (select, measure, combine, imports, config reading); `protocol-v4.json` (scope, method, limits); `write_protocol_lock.py` (metric A text, pins).
14. A path check showing that `--out .local/../scripts/x-paint` passes the out-folder test (path arithmetic only; nothing was created).
15. `validate_verdict.py`: checks that `verdict.json` parses and has exactly the required keys and types (result: "ok fix", 0 blocker, 2 major, 8 minor).

**Process slips.** None of these changed any file or touched held-out data.

- A few tiny JSON-reading Python runs early on lacked the `nice` prefix.
- Two short inline runs (`python -I -B -c`) had the working folder inside `.local/evaluation/pao-paint`. With `-I`, Python does not import from the working folder.
- One run used the system Python (`/usr/bin/python3 -c`, without `nice` and without `-I`) with the working folder in the rehearsal paint folder. That folder holds only data files (no `.py` files), so nothing could be imported from it; the run only read JSON.
- One inline run of the framework Python lacked `nice`.
- Several short reads used inline `python -I -B -c` (with `nice`) instead of a script file in this folder. For the final check of the two rehearsal paint receipts, the working folder was the rehearsal folder, which holds data only; with `-I`, Python does not import from the working folder.
- Text searches (grep) and folder listings covered only development and plan folders: `.local/evaluation/pao-paint`, `.local/evaluation/pao-accuracy/lock`, `.local/project` and the rehearsal folder (development sites only).

## 13. Files in this folder

- `README.md`: this report.
- `verdict.json`: the machine-readable verdict.
- `compare_rehearsal.py`, `compare_rehearsal.json`: rehearsal against development.
- `probe_dev_results.py`, `probe_dev_results.json`: development crossings.
- `probe_synthetic_metric_a.py`, `probe_synthetic_metric_a.json`: metric A on the stored validation rows.
- `probe_rule_variants.py`, `probe_rule_variants.json`: R0, R1, R2, power and transplant.
- `probe_joint_status.py`, `probe_joint_status.json`: shared errors and v4 status.
- `probe_untested_patterns.py`, `probe_untested_patterns.json`: stop line with bars, wide stop line.
- `probe_two_line.py`, `probe_two_line.json`: two-line crosswalk.
- `probe_material_band.py`, `probe_material_band.json` (K = 2 to 3), `probe_material_band_bright.json` (K = 4 to 5, clean), `probe_material_band_bright_dark.json` (K = 4 to 5, dark).
- `probe_faint_near_edge.py`, `probe_faint_near_edge.json`: faint first stripe.
- `probe_limit_line_gap.py`, `probe_limit_line_gap.json`: stop line close to the crosswalk.
- `env_versions.py`: prints Python and package versions next to paint v4's lock (no saved output).
- `check_readme_numbers.py`: re-reads the stored probe outputs to check the numbers quoted in this report (no saved output).
- `validate_verdict.py`: checks the keys and types of `verdict.json`.

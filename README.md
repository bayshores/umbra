# umbra

umbra is a local web app that shows how a declared what-if vehicle near a pedestrian crossing changes the straight line from a driver's eye to a person. You place a crossing, a vehicle and a person on a real recorded street image, move the vehicle, and compare the current and proposed positions side by side in one workspace. Each result is Blocked or Unknown with a reason; umbra never reports Clear.

The main scenes are Argoverse 2 recordings from Palo Alto, California: a camera image, LiDAR sweeps and the HD map of the same moment. One DrivingStereo frame is the original development scene. An optional assistant uses NVIDIA Nemotron through Nebius Token Factory to turn a plain-English what-if ("box truck 10 ft back, child") into declared sizes and a distance; umbra computes every result with its own geometry (see [NVIDIA Nemotron assistant](#nvidia-nemotron-assistant-nebius-token-factory)).

The runtime contains no procedural street, built-in actor dimension tables, fabricated unknown volume or demo response. A scene may ship a prepared, hash-pinned `suggestions.json` that pre-places the crossing, vehicle and person; each suggestion is labeled detected or declared default (on map-first scenes, mapped or annotated) and stays editable. Dimensions suggested by the assistant also appear as declared, editable what-if values. Missing assets and invalid inputs stop the comparison explicitly.

## Status and limitations

- **The accuracy test failed.** On October 16, 2026 a final test, with pass limits frozen before any of its data was opened, ran once on six unseen Palo Alto crossings. Placement accuracy failed at all six and the visibility gate did not pass. Later improvements were measured only on scenes already seen during development, so they are practice results, not proof of accuracy. See [Palo Alto accuracy results](#palo-alto-accuracy-results).
- **Geometry only.** A result says whether the straight line from the recorded camera position to a point at the person's height passes through the declared vehicle box (or the recorded vehicle's measured shape). It does not assess parking safety, whether a driver would recognise the person, crash risk, reaction time or legal compliance.
- **No Clear.** Missing depth, ground or calibration evidence stays Unknown with its reason; nothing is filled in. A line counts as Blocked only when it cuts at least 0.10 m into the vehicle; everything else is Unknown.
- **Development-grade references.** Distances are referenced to Argoverse 2 LiDAR and map data, or to DrivingStereo reference depth, not to a survey. Calibration and metric scale are provisional.
- **Declared what-ifs.** Vehicle and person sizes and positions that you enter, or that the assistant declares, are hypothetical. Suggested placements are labelled with their source and stay editable.
- **Scene data has its own terms.** `public/scenes/` holds four prepared Palo Alto scenes made from Argoverse 2 data (CC BY-NC-SA 4.0: non-commercial use only, with attribution, and adaptations under the same license) and one DrivingStereo scene (MIT). Each folder names its terms in `ATTRIBUTION.md`; see [public/scenes/README.md](public/scenes/README.md). The Apache-2.0 code license does not cover them.

## Requirements

- macOS or Linux. Tested on macOS on Apple silicon. Scene preparation publishes its output with file operations that are only supported on macOS and Linux.
- Node.js `^20.19.0` or `>=22.12.0` (required by `@vitejs/plugin-react` 5.0.4), with npm. Tested with Node.js 26.9.0.
- Python 3.13.5 for scene preparation, the Argoverse 2 tools and the Python tests; Python 3.12.0 for the optional RF-DETR vehicle detector. Install each requirements file into its own virtual environment: their NumPy, Pillow and SciPy pins differ.
- A Nebius Token Factory API key, only for the optional assistant.

## Run the app

```sh
npm install
npm run dev -- --host 127.0.0.1 --port 4173
```

Open http://127.0.0.1:4173. The app opens the first scene listed in `public/scenes/index.json`; the scene menu in the header lists the others. The frontend uses React, Vite, Three.js, the Overpass Mono font and Phosphor icons. No model or sponsor API call runs at startup; the optional assistant calls Nebius Token Factory only when you press Ask or Check distances. `npm run build` compiles the frontend without private configuration or datasets.

The local loader supports at most 1,000,000 points and 2,000,000 triangles per measured or estimated geometry, and 16,000,000 calibrated image pixels. These are allocation limits, not validated geographic coverage or performance guarantees. Geometry and depth streams must match their declared byte lengths and checksums; overlong streams are cancelled. A scene load has a shared 30-second deadline. Failure or leaving the view aborts its pending requests and body reads; Retry loading starts a new manual attempt. The deadline cannot preempt synchronous CPU validation, but a late result is rejected before exposure. Declared bounds must match the loaded geometry extrema within the existing 0.00002 m serialization tolerance, which is not physical measurement accuracy.

## Run the tests

| Command | What it checks | What it needs |
|---|---|---|
| `npm test` | Node tests: geometry, scene loading, suggestions, the assistant's guards and the evaluation scripts | the tracked scenes in `public/scenes/` |
| `npm run build` | the frontend build | nothing private |
| `npm run build:sites`, then `npm run test:sites` | packages the optional Sites worker (`worker/`) into `dist/` and tests it | `.openai/hosting.json` (below) and the tracked DrivingStereo scene |
| the Python commands below | scene preparation, Argoverse 2 tools, evaluation, cloud-job and sponsor adapters | one Python 3.13.5 environment |

`npm run build:sites` reads `.openai/hosting.json`, which is not tracked. For this worker it holds no database or storage binding; create it with exactly this content:

```json
{
  "d1": null,
  "r2": null
}
```

Packaging does not publish anything. A frontend-only build does not establish packaged-worker verification.

The DrivingStereo scene's placement suggestions (`suggestions.json`) are tracked. To make them again, use the RF-DETR detector, which uses Python 3.12.0 in its own environment. The weights (about 405 MB) are not installed by pip:

```sh
python3.12 -m venv .local/venv-detect
.local/venv-detect/bin/python -m pip install -r scripts/requirements-detection.txt
mkdir -p .local/venv-detect/rf_home
curl --fail --location --output .local/venv-detect/rf_home/rf-detr-medium.pth \
  https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth
shasum -a 256 .local/venv-detect/rf_home/rf-detr-medium.pth
```

The printed hash must be `749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8`; the detector refuses other weights. Then run the two commands in [Automatic placement suggestions](#automatic-placement-suggestions) with `<scene-id>` set to `2018-10-19-09-30-39_2018-10-19-09-40-40-716`.

The Python tests use the Argoverse 2 environment plus pandas, pytest and the ONNX packages (`onnx` with its pinned helpers from [scripts/requirements-model-inspection.txt](scripts/requirements-model-inspection.txt), except its NumPy pin, and `onnxruntime`):

```sh
python3.13 -m venv .local/venv-tests
.local/venv-tests/bin/python -m pip install -r scripts/california/requirements-av2.txt pandas==2.3.1 pytest==8.4.1 \
  onnx==1.23.1 protobuf==7.36.2 ml_dtypes==0.6.0 typing_extensions==4.16.0 onnxruntime==1.24.4
for d in tests tests/california tests/evaluation tests/cloud tests/sponsors; do
  .local/venv-tests/bin/python -B -m unittest discover -s "$d"
done
.local/venv-tests/bin/python -B -m pytest -p no:cacheprovider tests/evaluation/test_m7_child_preflight.py
```

**Results on a fresh clone.** The frozen method files of the Palo Alto evaluation (the site split, protocols, lock records and reviews; `git ls-files .local` lists all 30) are tracked at their original `.local/evaluation/` paths. Saved job files, download receipts and older development builds stay in the maintainer's ignored `.local/` folder. This was checked on a fresh clone of commit `b8c806f` on 2026-10-16, with exactly the environments above and nothing else copied in:

| Command | Result on a fresh clone |
|---|---|
| `npm test` | 308 pass, 0 fail, 22 skipped (the skipped tests read older Palo Alto development builds that stay in `.local/`) |
| `npm run build`, `npm run build:sites`, `npm run test:sites` | build succeeds; 5 site tests pass |
| `tests` | 56 run, all pass (8 skipped) |
| `tests/california` | 289 run, all pass (18 skipped) |
| `tests/evaluation` | 296 run, all pass |
| `tests/cloud` | 108 run, 4 errors, all from missing saved job files under `.local/evaluation/auto-placement-2026-10-11/` (30 skipped) |
| `tests/sponsors` | 22 run, all pass |
| the pytest file | 58 pass |

The app itself was opened from the same fresh clone: all five scenes load from the tracked files, and the console shows no errors.

Without the ONNX packages, `tests/test_onnx_inspection.py` cannot load and 14 more California and cloud tests report errors (measured on a fresh clone of `e785321`). Other tests that need local development data, a historical git commit or an optional package skip and say why. Some optional tools have their own test commands in the sections below.

## NVIDIA Nemotron assistant (Nebius Token Factory)

Once the crossing, vehicle and person are placed (by suggestion or by hand), the assistant row under the header accepts a plain-English what-if, such as "box truck 10 ft back, child". NVIDIA Nemotron 3.5 Lightning, called at runtime through the Nebius Token Factory chat API with tool calling, declares the vehicle size, person height and distance. umbra applies them to the normal fields and computes Blocked or Unknown with its own geometry. Nemotron then writes a short answer from those computed results.

Check distances asks Nemotron to pick your vehicle plus up to three common vehicle types. umbra tests each one locally at every 0.25 m of setback and shows where the declared vehicle stops blocking the line from the driver eye to the person's height point. The 15 ft and 20 ft marks are shown as reference. California Vehicle Code §22500(n) uses 20 ft for parking near crosswalks. umbra does not decide legal compliance. Click a row to apply that vehicle and distance.

Nemotron never chooses image points or coordinates and never decides visibility. Every value it declares stays visible and editable. Every computed result carries one phrase: "blocked", "not blocked by the vehicle (rest unknown)" only when the line misses the box and nothing else made it Unknown, or "unknown (reason)", such as "unknown (too close to call)". Nemotron is told to use these phrases word for word. Answers that use unsupported wording, numbers absent from the computed results, a number for the wrong quantity, or "not blocked" where the computed result says otherwise are replaced by the app's own computed line. Dragging, sliders and comparisons never call the model.

To enable it locally, export these before `npm run dev` (see `.env.example`):

```sh
export NEMOTRON_ENABLED=1
export NEMOTRON_BUDGET_USD=0.05
export TOKEN_FACTORY_KEY_FILE=/path/to/token-factory.key
```

The key file must hold only your Token Factory API key, end in `.key`, be owned by you with mode 600, and be neither a symlink nor a hard link. The local server reads it only for a request and never sends it to the browser. The assistant route answers only loopback requests from this machine. The local server fixes the model, prompts, tools and a 600-token output cap. It allows one call at a time with no retries. It records each reservation and settled cost in `nemotron-usage.jsonl` next to the key file, so every checkout using that key shares one spending cap that survives restarts. The hosted Sites worker keeps the assistant off until a durable shared spending limit exists.

The scope is local placement and visibility comparison. There are no packet or export controls and no download endpoint. Scene assets retain provenance and integrity checks for the measured pipeline. Unknown routes stay 404.

## Use the workspace

Use Crossing, Vehicle and Person in the header to select the object you are placing. Choose the crossing edge, the current vehicle's front, and the pedestrian's ground point on the recorded image. Only the selected object's dimension controls appear, beside its placement on desktop and below the image on mobile. Each selection must have actual reference depth and lie near the estimated road plane. Dimmed pixels lack depth. A pending crosshair identifies a measured road pixel; an X identifies an unsupported selection, and Set point remains disabled. Failed selections retain the committed points. Pixel coordinates and keyboard arrows provide exact selection when the displayed image is smaller than its source pixels; there is no nearest-pixel substitution.

Enter vehicle width, height and length, and target height in meters. These describe declared what-if actors, not detected objects. Compare positions becomes available when all required inputs are valid. It initializes the proposal at the annotated current position and the driver at the recorded camera, in the same workspace. Drag the circular split grip, or use its keyboard arrows/Home/End, to reveal current and proposed positions through one shared calibrated camera. Move the proposed vehicle with the top-down drag, slider or numeric input. The driver slider moves both scenarios along the annotated parking axis and remains behind the target. Bounds come from the captured geometry and annotation. Edit placements returns to the existing annotations and dimensions. The top-down map identifies current/proposed vehicles, person and driver with object pictograms; the proposed vehicle has a filled sea-glass marker. No sightline ray or letter-marker key is drawn.

Placements retain their selected measured 3D coordinates, including local height, rather than flattening them onto the estimated road plane. The vehicle's front midpoint and person's foot therefore remain registered to their selected pixels at the recorded camera. Parking distances and movement use a horizontal axis; the proposed vehicle keeps the measured starting height. This is a declared pose, not validated contact across the vehicle footprint or tracking of terrain at its proposed position. Pixel registration does not establish physical depth accuracy.

## Automatic placement suggestions

When a scene ships `suggestions.json`, it opens already set up in the comparison view. The status line under the header says what was suggested and where it came from, for example `Suggested · crosswalk detected ±0.5 m · SUV at 20 ft (default, no parked vehicle) · child 1.1 m (default)`. Choose Edit placements, then click the image, use the arrow keys or type pixel values to move a point, and type to change sizes; drag or type to move the proposed vehicle. Edited items lose their Suggested label, and Reset to suggested restores the prepared setup.

The suggestions are prepared once per scene, never guessed at runtime:

- `scripts/suggest-placements.py` (numpy and Pillow, the preparation environment) finds the right curb as the measured road-to-sidewalk height step and the zebra crosswalk as periodic bright bars on a bird's-eye view resampled on the depth-registered ground plane. The crossing point sits on the crosswalk's near edge in the curb-side lane, with about ±0.5 m uncertainty along the road because the sparse reference depth clusters about 1 m apart at that distance.
- `scripts/detect-vehicles.py` runs RF-DETR Medium (`rfdetr==1.11.2`, Apache-2.0, weights sha256 `749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8`) on the recorded image and records cars, trucks and buses with their measured-depth ground positions; `suggest-placements.py` then keeps only those parked in the curb lane between the camera and the crossing. Install its pinned environment from `scripts/requirements-detection.txt` in a separate virtual environment.
- When no parked vehicle is detected, the vehicle is a declared default SUV (1.9 × 1.8 × 4.9 m) 20 ft (6.096 m) from the crossing, the distance California Vehicle Code §22500(n) uses near crosswalks. The person is a declared default child target (1.1 m) at the curb end of the crossing. Defaults are data in `suggestions.json`, not code, and are always labeled.
- Every suggested pixel must have measured depth and pass the same road-surface check as a click; anything that cannot be established is written as unknown and left unplaced. The loader verifies the catalog's sha256 for `suggestions.json` and its recorded input hashes before applying it.

```sh
.local/venv-detect/bin/python -I scripts/detect-vehicles.py public/scenes/<scene-id>
.local/venv-preparation/bin/python -I scripts/suggest-placements.py public/scenes/<scene-id> --detections public/scenes/<scene-id>/detections.json
```

When a second detector's run is supplied with `--cross-check`, `suggestions.json` records it as an optional `crossCheck` block with the same fields as `detection`. Both blocks record their own `curbLaneParked` count, computed by the same rule (eligible curb-lane candidates between the camera and the crossing); the script refuses a cross-check whose detections files use a different curb-lane rule or curb, and writes nothing when the two counts differ. The cross-check never sets the suggestion, and the app names it in the vehicle's hover text. The loader also rejects a cross-check with the same weights or detections file as the detector, or a different `curbLaneParked` count. The maintainer's locally prepared DrivingStereo suggestions use RF-DETR only and carry no cross-check; their `suggestions.json` and `detections.json` were generated by the two scripts as committed in `aa581fa` (the `generator.sha256` values match that version) and were not regenerated after the cross-check support was added. Job tooling for NVIDIA NV-Grounding-DINO as an additional detector (cross-checked against RF-DETR) is in `scripts/cloud`, but that job never ran and no NV-Grounding-DINO outputs exist (see its README).

The suggestion script writes the catalog entry's `suggestions` hash and refuses to overwrite an existing file unless `--replace` is given. Its unit tests run with `.local/venv-preparation/bin/python -I -B -m unittest discover -s tests -p 'test_suggest_placements.py'`. These suggestions were checked on one scene against a hand test annotation, not against independent ground truth.

## Map-first scenes (Argoverse 2, local only)

An indexed LiDAR scene stays read-only unless its pinned `scene.json` declares `measurements.groundPolicy: "mappedGround"` with a `ground.json` descriptor that the catalog entry pins as well (`ground: {file, sha256}`). The loader then verifies `ground.json` and its `terrain.bin` raster by size and sha256, checks the city frame, drivable areas, ego lane, annotated boxes and LiDAR support rows, and rejects the whole scene on any mismatch. There is no read-only fallback or partial ground.

With a schema 3 ground layer, a clicked or suggested pixel maps to the first hit of its camera ray with the publisher ground raster inside the mapped drivable areas, within the ground layer's range. A hit more than 30 m from the camera (horizontal) is refused: "Too far to place (over 30 m)." Each point's along-ray uncertainty is the ground layer's height band (fitted on that development scene's LiDAR ground returns, in-sample) divided by the sine of the ray's angle below horizontal. Each point carries a support class: `observed nearby` when a camera-visible LiDAR return agrees with the map within 0.5 m, otherwise `mapped, not observed`. A point is `hidden` when its pixel shows an annotated object and the ground point lies behind it; hidden points are drawn as hollow markers. Missing map height, rays that never reach mapped ground, and hits outside the drivable areas stay Unknown with their reason. The comparison's evidence key lists the support of every placed point, suggested or edited, and Nemotron's facts carry it too. A declared vehicle runs along the mapped ego-lane axis through its own front; the annotated vehicle described below keeps its own heading. Either one stays on contiguous mapped drivable ground between the crossing and the recorded camera. Mapped ground is placement support only: results remain Blocked or Unknown and never Clear.

A schema 4 ground layer (`measured-ground-v1`, written by `scripts/california/measured_ground.py`) replaces the publisher raster with a measured surface: a 0.5 m grid of ground heights fitted from other LiDAR sweeps of the same log. The sweep the scene's evaluation scores against is never used. Sweeps recorded within 0.25 s of it, or with its exact bytes (sha256), are refused, and the loader rejects a layer that does not name that sweep's sha256. Every fourth remaining sweep is held back for calibration and is not used for the surface. Grid points without measured LiDAR support stay holes; on the five development scenes 56–76% of the 64 m grid is holes. The pixel's camera ray is followed through the triangulated surface. If it reaches a hole before it reaches ground, the point is Unknown ("No independently measured ground support along this ray."). There is no fallback to the publisher raster or to any filled-in surface. The 30 m limit and the drivable-area rule still apply. Each point's uncertainty is a horizontal distance looked up by its range (bands 0–10, 10–12, 12–15, 15–18, 18–20, 20–22, 22–25 and 25–30 m): the 97.5th percentile of the click-to-ground errors measured on the held-back sweeps, with at least 100 measurements per band, pooled from neighbouring range bands when a band has fewer of its own (the layer records each band's own count and pooled range). On the development scenes the 25–30 m band is 1.46–2.01 m wide. A placed point on this layer is always `observed nearby`. An untouched mapped crossing keeps its exact mapped source point; it is Unknown when that point is a hole. The support, calibration and scored sweeps come from the same log, so they are not fully independent. See [Palo Alto accuracy results](#palo-alto-accuracy-results).

A `measured-ground-v2` layer (also schema 4) keeps every `measured-ground-v1` height exactly (tier 0) and fills only v1 holes (tier 1) with the same unchanged plane fit, using the other sweeps of the whole log outside 0.25 s of the scored sweep; every fourth of those sweeps is held back for tier-1 calibration. No height is copied, interpolated or smoothed into a hole; whatever is still unsupported stays a hole. Each vertex records its tier and a triangle takes its highest vertex tier. Each tier has its own range bands; a hit on a tier with no band at that range is Unknown ("No click-error calibration for measured ground tier 1 at this range."). A tier-1 point is labelled `measured at another moment of the same recording`. The v2 layer pins the exact v1 `ground.json` by sha256; the scene folder carries it as `tier-0-ground.json`, and the loader rejects the v2 layer unless every tier-0 height, the grid, the fit, the sweep partition and the tier-0 bands equal that v1 file and tier 1 fills only its holes. On both schema 4 layers a vehicle's movement run ends only at the mapped drivable area or the measured grid's window, not at a hole; the moved vehicle keeps its starting front height, and holes still make ground points, anchors and the sightline checks that need ground Unknown. When a suggested point cannot be placed, the app names that role and the reason. On the five development scenes v2 removes 78–100% of the drivable holes 5–30 m from the camera.

Schema 3 `suggestions.json` (protocol v3, from `scripts/california/suggest_av2_placements.py`) with a schema 3 `ground.json` (placement protocol v3.4, which adds the recorded vehicle's measured shape, `carShape`, built by the shape-v2 rule; schema 1 and 2 layers, and schema 3 layers with a shape-v1 shape from protocol v3.3, are refused) is validated by re-running the runtime rule at every stored exact sub-pixel position (`pixelUV`); earlier schemas are refused, and so are protocol v3 layers (same schema numbers, superseded by v3.1), by their protocol hash. While a suggested point's integer pixel is unchanged, the app evaluates that exact position (the integer pixel alone can land up to 1 m away near the horizon). Positions that are clicked, typed or declared are whole pixels: a decimal pixel such as 374.5 is refused with "Select a pixel inside the recorded image.", so only an untouched suggestion uses a sub-pixel position. The crosswalk chip shows the crossing's own error, the mapped edge with its paint offset or the offset's status ("crosswalk mapped · paint offset pending measurement"), never the along-ray figure. An annotated stationary vehicle is labelled stationary, with no claim about why it stopped; its box is the average of all its annotations in the log. While its front pixel and size still hold the suggestion and no declaration has replaced the vehicle, the current vehicle is the annotated box itself (centre, horizontal heading and dimensions; its sub-degree pitch and roll are not used), in the comparison, previews and Nemotron's facts. The proposed what-if and the Distance check's "your vehicle" row use that same box moved along the lane axis, so the comparison opens with both sides showing the same box and the same result. For this box, the setback is the distance along the lane axis from the crossing point to the centre of the box's front face. Nemotron's facts give that setback and how far the box is turned from the lane (`turnedFromLane_deg`). The box becomes a declared box on the lane axis, labelled "Declared", once its front is moved more than 2 pixels on either image axis, its size changed, or a Nemotron declaration or another vehicle's distance row replaces it, even at the same sizes. Reset to suggested restores it. So does the checked vehicle's own distance row, when the check ran on the annotated box. Near the horizon one image pixel can move a ground point by about a metre: the suggested crossing and front pixels are 3 px apart. So a 1–2 px nudge of the front keeps the annotated box and its result; the nudged pixel only marks the vehicle. A declared or resized vehicle that replaces the annotated one while the front pixel is still within that nudge keeps the annotated vehicle's front point (protocol v3.1): the front pixel's own ray can be blocked by the raster short of the real front (on the 8c scene by 1.03 m, which had turned an Unknown into a false Blocked). A front moved further declares a box at the new front, or is refused with the measured gap when that front lands past the crossing or within 0.08 m of it. That new front is placed by its own pixel ray, so beyond the 2 px nudge the raster shadow still applies: on the 8c scene a front moved 3 px (pixel 377, 191) is a declared box 1.41 m from the crossing and Blocked, while a 2 px nudge (376, 193) keeps the stationary vehicle 0.42 m from it (Unknown). This 1 m jump at the nudge boundary leans toward a false Blocked and is still open. A one-pixel edit of the crossing keeps the annotated box, or is refused with the gap when the box's nearest front corner would be past the crossing. Because the box is turned slightly from the lane, one front corner is nearer the crossing than the front-face centre; the slider and the distance check start where that corner is the road tolerance from the crossing, and a what-if whose corner would reach the crossing is refused. On these scenes Nemotron's answers and labels may not use any word built on "park": such an answer is replaced by the app's computed line, and such a label is refused so the model renames it. Argoverse 2 data is CC BY-NC-SA 4.0 ((c) 2021 Argo AI, LLC); the app shows that attribution in the evidence key, links the license and gives the change notice on hover and to screen readers. The Distance check's Vehicle Code note is not shown on these scenes. These scenes are not shipped in `public/` or the hosted build.

## Palo Alto accuracy results

umbra has not passed its accuracy test. All numbers below are referenced to Argoverse 2 LiDAR and map data; they are development-grade, not surveyed. They describe geometric placement and visibility only, not parking safety, recognition, crash risk or legal compliance.

**Final held-out test (failed).** On October 16, 2026 a protocol with frozen pass limits, registered before any of its sensor data was opened, ran once on six Palo Alto crossings not used during development. The visibility gate did not pass. Placement accuracy (B) failed at all six. B requires, for clicked road pixels 12–25 m away: 95th-percentile click-to-ground error ≤ 0.76 m, the shown uncertainty band containing the LiDAR reference point at least 95% of the time, and every band ≤ ±0.76 m. Only one crossing produced a scene with a measured vehicle shape. There, B (p95 1.03 m), the crossing-point consistency check (0.147 m; limit 0.05 m), the vehicle-outline check against independent image labels (median 3.07 px; limit 3 px) and one honesty check (three wrong-quantity sentences accepted) failed. No result there was a false Blocked against the labels, but only 41 of 1,748 label rows could be resolved; the vehicle-shape cross-check passed, and useful coverage passed (17 of 33 reference-blocked child rows reported Blocked). Two of the three Blocked-test crossings produced no usable camera frame and count as misses. At two other crossings the evaluation's baseline step crashed and counts as failed. These results are final and were not rerun.

**After the test (practice results only).** The following were measured only on five crossings already seen during development (the scenes from the held-out test now count as development scenes), so they cannot demonstrate accuracy:

- The baseline crash and the honesty check's rounding bug were fixed.
- A crossing-anchor rule limited to the finite mapped crosswalk edge found no new usable camera frames.
- A metadata search of all 1,000 Argoverse 2 sensor logs found no unused Palo Alto crossing that qualifies as a Blocked test, so a second Blocked test was not possible.
- B still fails at all five scenes on both ground layers. Original raster: p95 0.76–1.18 m, band coverage 86–97%, widest band ±0.73–1.83 m. Measured layer (schema 4, above): p95 0.71–0.98 m, coverage 77–98% of placed points, widest band ±0.73–2.01 m. The recording camera is about 1.5 m above the road, so at 22 m a ground-height error grows about 15 times along the road: a 5 cm error moves the point about 0.7 m.
- On the measured layer (v1), three of the five scenes first lost their suggested setup (a suggested point fell in a support hole, or the suggested vehicle extended beyond the captured road), so the vehicle checks could not be scored there.
- `measured-ground-v2` (above) and the map-decides-road-end rule restore the suggested setup at one of those three (c780). At 8c the crosswalk and vehicle-front points sit in front of a real stationary vehicle and no sweep of the whole log has a ground return within 0.5 m, so they stay Unknown; at 88f47a10 the crossing is now supported but has no tier-1 band, and the vehicle-front and person points stay unmeasured. B still fails at all five scenes on v2: p95 0.71–0.98 m, coverage 77–98% of placed points, widest band ±0.73–2.01 m (tier-0 points are unchanged from v1; tier-1 bands exist at only two scenes and widen the largest fd5c band from ±1.38 to ±1.98 m). Of the vehicle checks, c780 scores F, F0 and I as passing on v2 and fails useful coverage (H); 8c and 88f47a10 stay unscored and fail H.
- A second held-out test of ground placement, paint position and crossing position on three unopened crossings was cancelled before it was registered. No pass limit was changed. An option that refuses any point whose band exceeds ±0.76 m was not adopted; it would pass B at only two of the five scenes.

Paint measurements did not produce a scored paint-position (A) result. Browser speed was not measured. The image-label checks were measured only at the one held-out crossing above.

## Measured evidence

The prepared scene catalog and assets live in `public/scenes/`. The prepared DrivingStereo development frame (`2018-10-19-09-30-39_2018-10-19-09-40-40-716`) contains 72,707 points and 11,990 triangles after an 80 m depth cutoff. Supported reference covers 20.7% of image pixels. Triangles connect only fully supported adjacent pixels whose three edges are at most 0.4 m. Missing regions stay dark; moved cameras expose additional unknown surfaces. RGB colors come from the recorded left image.

The road plane is estimated by RANSAC and SVD from supported points in a declared lower-central image region. This frame's fitted camera height is 1.805 m, road inlier fraction 96.3%, and median road-fit residual 0.0234 m. Residual is not an accuracy guarantee or surveyed ground truth. A failed support gate stops preprocessing.

Calibration uses the full-resolution rectified projection scaled to the actual image dimensions. The supplied half-resolution right projection retains an unscaled translation column and implies a doubled baseline. The [author paper](https://yangguorun.github.io/papers/cvpr19_drivingstereo.pdf) specifies about 54 cm, agreeing with the full calibration; a fixed 21-pair correspondence check also supports the chosen convention. The [official dataset](https://drivingstereo-dataset.github.io/) defines half-size depth as uint16 divided by 256, zero invalid. The app provisionally interprets it as camera-axis Z; exact export semantics and independent metric precision remain unverified. The reference combines temporally fused LiDAR with stereo-model-guided filtering, as described in the author paper. The 1/256 encoding step is not a physical accuracy bound; exact scan provenance, export resampling and per-point uncertainty remain unresolved. Original calibrations and numerical checks are retained in the ignored acquisition folder. Pixel agreement does not validate visibility or safety.

Blocked means the eye-to-target segment cuts at least 0.10 m into the complete declared vehicle cuboid: the box would have to shrink by 0.10 m on every side before the segment missed it. Otherwise the result is Unknown with a reason. A shallower cut is "too close to call" (margin set on October 12, 2026; it is not yet justified by a locked error budget). The other reasons are a person or driver eye inside the vehicle box. On map-first scenes they also include a person or eye inside a real annotated object, and a vehicle box that overlaps one. The stationary vehicle's own annotation is not counted as such an object. These rules only turn Blocked into Unknown; nothing becomes Clear. On map-first scenes the real recorded stationary vehicle, while it stays in its annotated pose, also carries a LiDAR-measured shape (shape-v2, `scripts/california/car_shape.py`): a 2.5D height field on a grid of cells at most 0.10 m wide over its box footprint, built from public LiDAR sweeps of its log (sweeps within 30 m, one kept each time the recording car moved at least 0.25 m, at most 40, split alternately into shape and check sweeps; each cell's top is the second-highest of the per-sweep maximum heights; a cell without enough returns counts as height 0, and everything below a top counts as solid, windows included). A line that cuts at least 0.10 m into that box is Blocked only when it also runs at least 0.10 m deep inside the measured shape; otherwise it is Unknown, "vehicle height not proven". A shape that cannot be built leaves every such line Unknown with the reason; it never falls back to the box. The shape can only turn a Blocked into Unknown. Declared, resized and replacement vehicles keep the plain box, and the views still draw the box. The comparison, the Distance check rows and Nemotron's facts all use the same rule (`assessSightline`) and expose the reason and the signed gap between the line and the box. The driver eye is the recording camera's position and height, about 1.47 m above the mapped ground on the Argoverse 2 scene, not a seated driver's eye. Typed and declared sizes share one range: vehicle width 0.5–3.5 m, height 0.5–4.5 m, length 1–20 m, and person 0.3–2.5 m. Numbers must be plain decimals, so "0x2" is refused, never read as 2. A setback below the slider minimum is refused. A contact with a reconstructed reference triangle remains Unknown: adjacent sparse returns constrain its vertices, but do not establish an opaque continuous surface between them. Triangle contact, including coplanar/grazing cases, remains tested separately from this evidence policy. The calculation uses floating-point determinant and plane tolerances rather than exact arithmetic; rounding constructed near-coplanar points can affect contact. This numerical tolerance is not a sensor uncertainty model. Vehicle orientation must be a horizontal unit axis. Camera transforms must be proper orthonormal rotations and agree with the renderer's pose; invalid calibration is rejected without repair.

The low-level `triangleHit` helper takes ordinary arrays of three finite numbers. It rejects typed arrays, extra components, sparse holes and nonfinite coordinates. The runtime copies loader-validated measurement coordinates into dense triples and uses the same calculation without repeating those checks for every triangle.

The current sparse-depth pipeline does not produce Clear. Its previous pixel-cone rule could infer empty space around a sampled ray without observing that space. The angular-depth coverage calculation is retained as a diagnostic, and any remaining sightline is Unknown until a sensor-specific free-space footprint and operating resolution are independently validated. Missing depth, out-of-frame regions and beyond-observed-surface space remain unknown. Useful validated clear-space coverage is unfinished; returning Unknown everywhere is not the project's completion criterion.

This compares one target point under geometric assumptions. It does not assess safety, crash risk, recognition, reaction time or regulatory compliance. Parking-ranking robustness, uncertainty sensitivity, independent scenes and physical reconstruction accuracy remain pending.

## Development evaluation

The optional CPU stereo study uses the existing [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html) implementation with fixed parameters on actual downloaded DrivingStereo pairs. It writes raw disparity, both provisional calibration conventions, error/coverage denominators, source/output hashes and environment details. It is development evidence from one sequence, not a California/held-out visibility validation or sponsor-model output. It needs NumPy and OpenCV.

```sh
python3 scripts/evaluate-stereo-baseline.py --self-check
python3 scripts/evaluate-stereo-baseline.py --count 5 --output .local/evaluation/stereo-baseline/my-run
node scripts/benchmark-geometry.mjs
```

The stereo output directory must be new so a run cannot silently overwrite prior evidence. Geometry benchmark inputs are isolated test annotations on the real prepared scene; its report measures one-scene Node CPU queries, not browser FPS or model/GPU inference.

`scripts/evaluate-epipolar-alignment.py` checks fixed SIFT correspondences across 21 actual development pairs, retaining descriptor rejects, F/H masks and regional support. The current study finds a small vertical offset, with no saved feature matches in the bottom-central road region where depth errors are severe. These automatic correspondences do not establish metric calibration or independent accuracy.

`scripts/evaluate-stereo-offset.py` reuses the saved five-frame baseline and tests one frozen right-image vertical translation plus its opposite-sign control. It records interpolation/support masks and paired as well as full-reference denominators. Both scripts have `--self-check`; their full runs require the acquired source/evaluation artifacts and refuse to overwrite existing reports. The offset diagnostic is development evidence; the app does not apply a calibration warp from it.

The optional offline `scripts/inspect-onnx-model.py` checks an authorized local model's SHA-256 and byte ceiling before parsing it with ONNX. It reports declared input/output types/shapes, opsets, operators, local functions and external tensor references. It never opens external tensor sidecars or runs inference. Embedded structural-check success exits 0; detected external references produce a blocked report and exit 2; invalid input/checker/output failures exit 1 and preserve the prior report. Graph structure does not establish normalization, runtime compatibility or accuracy. Tests use tiny parser fixtures, not NVIDIA outputs.

`scripts/prepare-stereo-crop.py` takes explicit left/right RGB images, raw uint16 reference PNG, calibration, integer crop, depth divisor and a new output directory. It preserves decoded pixels with no resize/fill, loads projection at the original image size and translates both principal points by the same crop. Calibration is parsed from its verified byte snapshot. It records hashes and projected right-image boundary exclusions without removing those reference samples. This prepares coordinates, not model tensors or validated inference; axial-Z/divisor remain declared assumptions. Run `python3 -m unittest discover -s tests -p test_stereo_crop.py -v` for analytic fixtures. Exclusive staged publication currently supports macOS/Linux; unsupported platforms reject, and power-loss durability is unverified.

Install its separate Python dependencies from `scripts/requirements-model-inspection.txt` into a development virtual environment, then run `python -m unittest discover -s tests -p test_onnx_inspection.py -v` with that interpreter. Invocation requires a local model path, `--expected-sha256`, `--max-bytes` and `--output`; the output directory must already exist. Use the published hash and exact file size for the chosen revision. This tool does not download models or authorize their use. NVIDIA's [C-FoundationStereo-S card](https://huggingface.co/nvidia/c-foundationstereo-s) says downloading accepts its [license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The pinned fixed320×736 v2 artifact has been acquired and inspected, and one bounded local CPU experiment completed with real raw disparity. Its [contract and execution adapter](scripts/sponsors/README.md) use version-specific NVIDIA preprocessing. The application can now consume a source-bound cached estimate for rendering alongside the unchanged measured assets. Local model execution and application consumption have been verified for one development pair, and one Nebius AI Cloud run is described below; physical accuracy remains open.

## Attach an authorized cached NVIDIA reconstruction

`scripts/attach-nvidia-reconstruction.py` adds separate predicted point/color/triangle buffers to a new prepared scene directory. It verifies the actual pinned model bytes, registered prediction, original producer/crop/source/calibration identities and measured geometry. Existing output directories are retained. It neither downloads weights nor runs inference or a sponsor API.

```sh
python3 scripts/attach-nvidia-reconstruction.py \
  --scene /path/to/measured-scene/scene.json \
  --prediction /path/to/registered-prediction.json \
  --prediction-sha256 <verified-registered-manifest-sha256> \
  --model /path/to/authorized-pinned-model.onnx \
  --output /path/to/new-scene-directory
```

Use the existing preparation environment and the registered producer/source artifacts described in [the evaluation adapter](scripts/evaluation/README.md). The prediction manifest must retain its referenced source files and original inference receipts. Publication into the local scene catalog is a separate deliberate operation; the attachment tool does not edit a catalog or replace its input scene.

The current development scene renders 159,495 estimated points and 257,733 triangles from the genuine C-FoundationStereo-S output. Display support uses the existing 80 m range, nominal right-pixel bounds within the original crop and the existing 0.4 m adjacency edge limit. These are visualization limits, not model confidence, observed correspondence or validated physical surfaces. Raw predictions and full-reference diagnostics retain their outliers. Moving the camera reveals holes and stretched surfaces; no hidden filling is applied.

Measured depth, measured geometry, actor anchors, movement limits and visibility queries remain unchanged. The frontend labels the estimate separately from measured anchor support. Missing or corrupted declared estimate assets stop scene loading; measured geometry is not silently substituted after a failure. Vehicle dragging, driver movement and ordinary comparisons only use local assets and calculations. On October 10, 2026 the same pinned C-FoundationStereo-S model also ran once as a Nebius AI Cloud Serverless Job on CPU for one development crop. Its output was retrieved, checked and attached through `scripts/california/attach_av2_prediction.py --direct-native` with the Job's control record. Neither run establishes independent accuracy or California coverage.

## Reproduce scene preparation

The Palo Alto (Argoverse 2) scenes are built with the [California offline tools](scripts/california/README.md), which download public AV2 sensor logs, assemble indexed LiDAR scenes without inventing dense depth, and add the measured ground layer, the recorded vehicle's shape and placement suggestions. Physical validation remains incomplete (see [Status and limitations](#status-and-limitations)). The DrivingStereo scene below is prepared separately.

The source is the official [DrivingStereo dataset](https://drivingstereo-dataset.github.io/); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its license. Only the one prepared scene above is tracked; download the dataset yourself to prepare others. The maintainer's copy (500 matched image/depth triplets, calibrations, acquisition hashes and inspection records) lives in the ignored `.local/data/drivingstereo/`. The preparer copies the dataset license into each prepared scene and into `public/scenes/LICENSE`.

Install the verified preparation dependencies in a dedicated environment:

```sh
python3 -m venv .local/venv-preparation
.local/venv-preparation/bin/python -m pip install -r scripts/requirements-preparation.txt
```

For rectification callers, [remap_support.py](scripts/remap_support.py) selects a deterministic all-supported rectangle from the common boolean source-remap mask. `supported_rectangle(mask, crop_size, margin=1)` requires the requested crop to fit after the margin, rejects holes and missing usable support, and leaves the mask unchanged. It avoids relying on OpenCV's rounded integer ROI. The caller must establish the mask's source/calibration identity; source pixel support does not establish physical visibility or depth accuracy. Its focused checks run with `.local/venv-preparation/bin/python -B -m unittest discover -s tests -p test_remap_support.py -v`.

The data root needs these files, where `<frame>` is the complete sequence/frame identifier and `<sequence>` is its portion before the underscore:

```text
extracted/left-image-half-size/<frame>.jpg
extracted/right-image-half-size/<frame>.jpg
extracted/depth-map-half-size/<frame>.png
calibration/<sequence>-full-reference.txt
LICENSE
```

Use the sunny half-size left/right/depth archives and the matching full-resolution training calibration from the official download page. Copy the full calibration unchanged to the named `-full-reference.txt` path; the preparer scales its projection to the actual image dimensions. Keep the original file and dataset license. The supplied half-size calibration has the translation discrepancy described above and cannot be substituted by renaming it.

```sh
.local/venv-preparation/bin/python scripts/prepare-scene.py \
  --data-root .local/data/drivingstereo \
  --frame 2018-10-19-09-30-39_2018-10-19-09-40-40-716 \
  --output public/scenes \
  --road-tolerance 0.08 --max-depth 80 --mesh-edge-limit 0.4
```

`--output public/scenes` writes the scene folder and adds it to the app's catalog `public/scenes/index.json`; this is the scene `npm test` and `npm run test:sites` use. To replay a preparation without touching the app's catalog, give a new directory instead, for example `.local/evaluation/reprepared-scenes`.

The pinned preparation stack was installed in an isolated environment and replayed on Python 3.13.5/macOS arm64 across five frozen development frames. Four accepted scenes reproduced all eight non-metadata artifacts byte for byte, with identical camera, transform, processing and source metadata; only the recorded processing-code identities changed. The fifth retained its road-support rejection and left the catalog unchanged. This verifies the current preparer against prior prepared outputs, without establishing physical accuracy or other-platform equivalence.

Choose a new output directory for a replay; an existing scene is retained rather than overwritten. The current adapter accepts finite horizontal rectified matrices with positive focal lengths, a left-origin projection, positive baseline and matching source sizes/fx/fy/cy. Unsupported projection terms and malformed size fields stop processing. This checks the supported matrix form, not physical calibration or exact image alignment.

Preparation reads each input file once and uses those captured bytes for decoding, calibration, source hashes and copied images/license. A later edit to a source path cannot substitute different bytes into the prepared output. Keep a stable matched acquisition during input capture: these separate file reads do not establish synchronization across files or sensors.

It creates calibrated points, supported triangles, a depth buffer, support mask, source/code hashes and per-scene license. It stages the complete scene, catalog and any new shared license before committing the catalog with an atomic local replacement. Handled errors before that commit roll back this invocation's additions; an interruption immediately after a successful commit retains the complete scene. This assumes one writer and private staging paths. Cleanup failures, concurrent writers, process death and power-loss recovery are unproven. A nonzero exit can occur after a successful commit, so inspect the catalog and scene before retrying; existing scenes are retained rather than overwritten. The frontend reads the generated catalog rather than selecting a hardcoded frame. Tests contain explicitly isolated annotations on actual source pixels; they do not supply runtime defaults.

Preparation records the three processing source files (`prepare-scene.py`, `rectified_calibration.py`, `scene_surface.py`) observed after imports and rechecks them immediately before publication; detected changes or unreadable files stop the job. These hashes do not authenticate already loaded bytecode, dependencies or the complete runtime. Reproduce with fixed source files and the pinned preparation environment; changes restored between observations are not detected.

```sh
.local/venv-preparation/bin/python -B -m unittest discover -s tests -p 'test_scene_preparation.py' -v
```

The Python checks exercise unsupported/malformed calibration and processing controls, preserve existing destinations, and reproduce the original geometry buffers from the real acquisition. A controlled mid-run mutation of isolated real-source copies also verifies that prepared hashes, calibration and copied files retain the captured input identity. Publication checks inject eight I/O failures and five post-operation Python interrupts, verifying rollback or retained committed output as appropriate. Their prior-scene fixture is prepared from genuine acquired raw data. If that optional raw acquisition is absent, the suite explicitly skips its real-data portion and still runs analytic/control checks.

## Licenses

umbra's own code is licensed under the [Apache License 2.0](LICENSE). Third-party software, fonts, icons, models, services and datasets, with their licenses, are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); full license texts are in [licenses/](licenses/).

- Argoverse 2 data is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), © 2021 Argo AI, LLC. No raw AV2 file is tracked in git. The four Palo Alto scenes in `public/scenes/av2-pao-*` are derived from AV2 sensor data (camera image, LiDAR points, ground surfaces, map geometry); each folder's `ATTRIBUTION.md` gives the credit, license, log id and changes. These scenes and the values derived from AV2 that appear in other tracked files (log and annotation identifiers, file hashes, measured numbers in tests and evaluation records) remain under CC BY-NC-SA 4.0: attribution, non-commercial use only, share-alike, and changes noted. The app shows the attribution and a change notice on every AV2 scene. The [Argoverse terms of use](https://www.argoverse.org/about.html#terms-of-use) also forbid trying to identify people in the data and using Argo's name, logo or trademarks without written permission (attribution excepted). An Argoverse maintainer confirmed on [argoverse/av2-api#340](https://github.com/argoverse/av2-api/issues/340) that this hackathon use is fine under those terms and that ShareAlike covers only the data, not the code.
- NVIDIA Nemotron 3.5 Lightning is used through the Nebius Token Factory API under the OpenMDW License Agreement 1.1 and the Token Factory terms of service; no model weights are included.
- DrivingStereo data is MIT-licensed; its website asks users to cite the CVPR 2019 paper by Yang et al. One prepared DrivingStereo scene is tracked with its license, citation and change notes.
- Dataset terms are separate from the license of umbra's own code.

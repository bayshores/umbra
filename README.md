# Umbra

A local, single-workspace parking visibility comparison using real DrivingStereo images, sparse reference-depth data and an explicitly estimated NVIDIA reconstruction. The runtime contains no procedural street, built-in actor dimension tables, fabricated unknown volume, or demo response. A scene may ship a prepared, hash-pinned `suggestions.json` that pre-places the crossing, vehicle and person; each suggestion is labeled detected or declared default (on local map-first scenes, mapped or annotated) and stays editable. Dimensions suggested by the optional NVIDIA Nemotron assistant also appear as declared, editable what-if values. Missing assets and invalid inputs stop the comparison explicitly.

## Run and verify

```sh
npm install
npm run dev -- --host 127.0.0.1 --port 4173
npm test
npm run build
```

The frontend uses React, Vite, Three.js, Geist Mono and Phosphor icons. No model or sponsor API calls run at startup; the optional Nemotron assistant calls Nebius Token Factory only when you press Ask or Check distances. `npm run build` compiles the frontend without private configuration or datasets. Prepared measurement assets are intentionally untracked; prepare authorized real sources below before the data-dependent tests or comparison. Missing assets fail explicitly.

The local loader supports at most 1,000,000 points and 2,000,000 triangles per measured or estimated geometry, and 16,000,000 calibrated image pixels. These are allocation limits, not validated geographic coverage or performance guarantees. Geometry and depth streams must match their declared byte lengths and checksums; overlong streams are cancelled. A scene load has a shared 30-second deadline. Failure or leaving the view aborts its pending requests and body reads; Retry loading starts a new manual attempt. The deadline cannot preempt synchronous CPU validation, but a late result is rejected before exposure. Declared bounds must match the loaded geometry extrema within the existing 0.00002 m serialization tolerance, which is not physical measurement accuracy.

For the existing configured Sites adapter, run `npm run build:sites` followed by `npm run test:sites`. Packaging requires the actual local `.openai/hosting.json`; it does not synthesize configuration or publish anything. A frontend-only build does not establish packaged-server verification.

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

When a second detector's run is supplied with `--cross-check`, `suggestions.json` records it as an optional `crossCheck` block with the same fields as `detection`. Both blocks record their own `curbLaneParked` count, computed by the same rule (eligible curb-lane candidates between the camera and the crossing); the script refuses a cross-check whose detections files use a different curb-lane rule or curb, and writes nothing when the two counts differ. The cross-check never sets the suggestion, and the app names it in the vehicle's hover text. The loader also rejects a cross-check with the same weights or detections file as the detector, or a different `curbLaneParked` count. The shipped suggestions use RF-DETR only and carry no cross-check; their `suggestions.json` and `detections.json` were generated by the two scripts as committed in `aa581fa` (the `generator.sha256` values match that version) and were not regenerated after the cross-check support was added. Job tooling for NVIDIA NV-Grounding-DINO as an additional detector (cross-checked against RF-DETR) is in `scripts/cloud`, but that job never ran and no NV-Grounding-DINO outputs exist (see its README).

The suggestion script writes the catalog entry's `suggestions` hash and refuses to overwrite an existing file unless `--replace` is given. Its unit tests run with `.local/venv-preparation/bin/python -I -B -m unittest discover -s tests -p 'test_suggest_placements.py'`. These suggestions were checked on one scene against a hand test annotation, not against independent ground truth.

## Map-first scenes (Argoverse 2, local only)

An indexed LiDAR scene stays read-only unless its pinned `scene.json` declares `measurements.groundPolicy: "mappedGround"` with a `ground.json` descriptor that the catalog entry pins as well (`ground: {file, sha256}`). The loader then verifies `ground.json` and its `terrain.bin` raster by size and sha256, checks the city frame, drivable areas, ego lane, annotated boxes and LiDAR support rows, and rejects the whole scene on any mismatch. There is no read-only fallback or partial ground.

A clicked or suggested pixel maps to the first hit of its camera ray with the publisher ground raster inside the mapped drivable areas, within the ground layer's range. A hit more than 30 m from the camera (horizontal) is refused: "Too far to place (over 30 m)." Each point's along-ray uncertainty is the ground layer's height band (fitted on that development scene's LiDAR ground returns, in-sample) divided by the sine of the ray's angle below horizontal. Each point carries a support class: `observed nearby` when a camera-visible LiDAR return agrees with the map within 0.5 m, otherwise `mapped, not observed`. A point is `hidden` when its pixel shows an annotated object and the ground point lies behind it; hidden points are drawn as hollow markers. Missing map height, rays that never reach mapped ground, and hits outside the drivable areas stay Unknown with their reason. The comparison's evidence key lists the support of every placed point, suggested or edited, and Nemotron's facts carry it too. A declared vehicle runs along the mapped ego-lane axis through its own front; the annotated vehicle described below keeps its own heading. Either one stays on contiguous mapped drivable ground between the crossing and the recorded camera. Mapped ground is placement support only: results remain Blocked or Unknown and never Clear.

Schema 3 `suggestions.json` (protocol v3, from `scripts/california/suggest_av2_placements.py`) with a schema 2 `ground.json` is validated by re-running the runtime rule at every stored exact sub-pixel position (`pixelUV`); earlier schemas are refused, and so are protocol v3 layers (same schema numbers, superseded by v3.1), by their protocol hash. While a suggested point's integer pixel is unchanged, the app evaluates that exact position (the integer pixel alone can land up to 1 m away near the horizon). Positions that are clicked, typed or declared are whole pixels: a decimal pixel such as 374.5 is refused with "Select a pixel inside the recorded image.", so only an untouched suggestion uses a sub-pixel position. The crosswalk chip shows the crossing's own error, the mapped edge with its paint offset or the offset's status ("crosswalk mapped · paint offset pending measurement"), never the along-ray figure. An annotated stationary vehicle is labelled stationary, with no claim about why it stopped; its box is the average of all its annotations in the log. While its front pixel and size still hold the suggestion and no declaration has replaced the vehicle, the current vehicle is the annotated box itself (centre, horizontal heading and dimensions; its sub-degree pitch and roll are not used), in the comparison, previews and Nemotron's facts. The proposed what-if and the Distance check's "your vehicle" row use that same box moved along the lane axis, so the comparison opens with both sides showing the same box and the same result. For this box, the setback is the distance along the lane axis from the crossing point to the centre of the box's front face. Nemotron's facts give that setback and how far the box is turned from the lane (`turnedFromLane_deg`). The box becomes a declared box on the lane axis, labelled "Declared", once its front is moved more than 2 pixels on either image axis, its size changed, or a Nemotron declaration or another vehicle's distance row replaces it, even at the same sizes. Reset to suggested restores it. So does the checked vehicle's own distance row, when the check ran on the annotated box. Near the horizon one image pixel can move a ground point by about a metre: the suggested crossing and front pixels are 3 px apart. So a 1–2 px nudge of the front keeps the annotated box and its result; the nudged pixel only marks the vehicle. A declared or resized vehicle that replaces the annotated one while the front pixel is still within that nudge keeps the annotated vehicle's front point (protocol v3.1): the front pixel's own ray can be blocked by the raster short of the real front (on the 8c scene by 1.03 m, which had turned an Unknown into a false Blocked). A front moved further declares a box at the new front, or is refused with the measured gap when that front lands past the crossing or within 0.08 m of it. That new front is placed by its own pixel ray, so beyond the 2 px nudge the raster shadow still applies: on the 8c scene a front moved 3 px (pixel 377, 191) is a declared box 1.41 m from the crossing and Blocked, while a 2 px nudge (376, 193) keeps the stationary vehicle 0.42 m from it (Unknown). This 1 m jump at the nudge boundary leans toward a false Blocked and is still open. A one-pixel edit of the crossing keeps the annotated box, or is refused with the gap when the box's nearest front corner would be past the crossing. Because the box is turned slightly from the lane, one front corner is nearer the crossing than the front-face centre; the slider and the distance check start where that corner is the road tolerance from the crossing, and a what-if whose corner would reach the crossing is refused. On these scenes Nemotron's answers and labels may not use any word built on "park": such an answer is replaced by the app's computed line, and such a label is refused so the model renames it. Argoverse 2 data is CC BY-NC-SA 4.0 ((c) 2021 Argo AI, LLC); the app shows that attribution in the evidence key, links the license and gives the change notice on hover and to screen readers. The Distance check's Vehicle Code note is not shown on these scenes. These scenes are not shipped in `public/` or the hosted build.

## NVIDIA Nemotron assistant (Nebius Token Factory)

Once the crossing, vehicle and person are placed (by suggestion or by hand), the assistant row under the header accepts a plain-English what-if, such as "box truck 10 ft back, child". NVIDIA Nemotron 3.5 Lightning, called at runtime through the Nebius Token Factory chat API with tool calling, declares the vehicle size, person height and distance. Umbra applies them to the normal fields and computes Blocked or Unknown with its own geometry. Nemotron then writes a short answer from those computed results.

Check distances asks Nemotron to pick your vehicle plus up to three common vehicle types. Umbra tests each one locally at every 0.25 m of setback and shows where the declared vehicle stops blocking the line from the driver eye to the person's height point. The 15 ft and 20 ft marks are shown as reference. California Vehicle Code §22500(n) uses 20 ft for parking near crosswalks. Umbra does not decide legal compliance. Click a row to apply that vehicle and distance.

Nemotron never chooses image points or coordinates and never decides visibility. Every value it declares stays visible and editable. Every computed result carries one phrase: "blocked", "not blocked by the vehicle (rest unknown)" only when the line misses the box and nothing else made it Unknown, or "unknown (reason)", such as "unknown (too close to call)". Nemotron is told to use these phrases word for word. Answers that use unsupported wording, numbers absent from the computed results, a number for the wrong quantity, or "not blocked" where the computed result says otherwise are replaced by the app's own computed line. Dragging, sliders and comparisons never call the model.

To enable it locally, export these before `npm run dev` (see `.env.example`):

```sh
export NEMOTRON_ENABLED=1
export NEMOTRON_BUDGET_USD=0.05
export TOKEN_FACTORY_KEY_FILE=/path/to/token-factory.key
```

The key file must hold only your Token Factory API key, end in `.key`, be owned by you with mode 600, and be neither a symlink nor a hard link. The local server reads it only for a request and never sends it to the browser. The assistant route answers only loopback requests from this machine. The local server fixes the model, prompts, tools and a 600-token output cap. It allows one call at a time with no retries. It records each reservation and settled cost in `nemotron-usage.jsonl` next to the key file, so every checkout using that key shares one spending cap that survives restarts. The hosted Sites worker keeps the assistant off until a durable shared spending limit exists.

The current scope is local placement and visibility comparison. Packet/export controls and their download endpoint were removed at Sean's request. Scene assets retain provenance and integrity checks for the measured pipeline. Unknown routes, including the former download endpoint, stay 404.

## Measured evidence

The prepared scene catalog and assets live in `public/scenes/`. The included frame contains 72,707 points and 11,990 triangles after an 80 m depth cutoff. Supported reference covers 20.7% of image pixels. Triangles connect only fully supported adjacent pixels whose three edges are at most 0.4 m. Missing regions stay dark; moved cameras expose additional unknown surfaces. RGB colors come from the recorded left image.

The road plane is estimated by RANSAC and SVD from supported points in a declared lower-central image region. This frame's fitted camera height is 1.805 m, road inlier fraction 96.3%, and median road-fit residual 0.0234 m. Residual is not an accuracy guarantee or surveyed ground truth. A failed support gate stops preprocessing.

Calibration uses the full-resolution rectified projection scaled to the actual image dimensions. The supplied half-resolution right projection retains an unscaled translation column and implies a doubled baseline. The [author paper](https://yangguorun.github.io/papers/cvpr19_drivingstereo.pdf) specifies about 54 cm, agreeing with the full calibration; a fixed 21-pair correspondence check also supports the chosen convention. The [official dataset](https://drivingstereo-dataset.github.io/) defines half-size depth as uint16 divided by 256, zero invalid. The app provisionally interprets it as camera-axis Z; exact export semantics and independent metric precision remain unverified. The reference combines temporally fused LiDAR with stereo-model-guided filtering, as described in the author paper. The1/256 encoding step is not a physical accuracy bound; exact scan provenance, export resampling and per-point uncertainty remain unresolved. Original calibrations and numerical checks are retained in the ignored acquisition folder. Pixel agreement does not validate visibility or safety.

Blocked means the eye-to-target segment cuts at least 0.10 m into the complete declared vehicle cuboid: the box would have to shrink by 0.10 m on every side before the segment missed it. Otherwise the result is Unknown with a reason. A shallower cut is "too close to call" (margin set on October 12, 2026; it is not yet justified by a locked error budget). The other reasons are a person or driver eye inside the vehicle box. On map-first scenes they also include a person or eye inside a real annotated object, and a vehicle box that overlaps one. The stationary vehicle's own annotation is not counted as such an object. These rules only turn Blocked into Unknown; nothing becomes Clear. The comparison, the Distance check rows and Nemotron's facts all use the same rule (`assessSightline`) and expose the reason and the signed gap between the line and the box. The driver eye is the recording camera's position and height, about 1.47 m above the mapped ground on the Argoverse 2 scene, not a seated driver's eye. Typed and declared sizes share one range: vehicle width 0.5–3.5 m, height 0.5–4.5 m, length 1–20 m, and person 0.3–2.5 m. Numbers must be plain decimals, so "0x2" is refused, never read as 2. A setback below the slider minimum is refused. A contact with a reconstructed reference triangle remains Unknown: adjacent sparse returns constrain its vertices, but do not establish an opaque continuous surface between them. Triangle contact, including coplanar/grazing cases, remains tested separately from this evidence policy. The calculation uses floating-point determinant and plane tolerances rather than exact arithmetic; rounding constructed near-coplanar points can affect contact. This numerical tolerance is not a sensor uncertainty model. Vehicle orientation must be a horizontal unit axis. Camera transforms must be proper orthonormal rotations and agree with the renderer's pose; invalid calibration is rejected without repair.

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

The optional offline `scripts/inspect-onnx-model.py` checks an authorized local model's SHA-256 and byte ceiling before parsing it with ONNX. It reports declared input/output types/shapes, opsets, operators, local functions and external tensor references. It never opens external tensor sidecars or runs inference. Embedded structural-check success exits0; detected external references produce a blocked report and exit2; invalid input/checker/output failures exit1 and preserve the prior report. Graph structure does not establish normalization, runtime compatibility or accuracy. Tests use tiny parser fixtures, not NVIDIA outputs.

`scripts/prepare-stereo-crop.py` takes explicit left/right RGB images, rawuint16 reference PNG, calibration, integer crop, depth divisor and a new output directory. It preserves decoded pixels with no resize/fill, loads projection at the original image size and translates both principal points by the same crop. Calibration is parsed from its verified byte snapshot. It records hashes and projected right-image boundary exclusions without removing those reference samples. This prepares coordinates, not model tensors or validated inference; axial-Z/divisor remain declared assumptions. Run `python3 -m unittest discover -s tests -p test_stereo_crop.py -v` for analytic fixtures. Exclusive staged publication currently supports macOS/Linux; unsupported platforms reject, and power-loss durability is unverified.

Install its separate Python dependencies from `scripts/requirements-model-inspection.txt` into a development virtual environment, then run `python -m unittest discover -s tests -p test_onnx_inspection.py -v` with that interpreter. Invocation requires a local model path, `--expected-sha256`, `--max-bytes` and `--output`; the output directory must already exist. Use the published hash and exact file size for the chosen revision. This tool does not download models or authorize their use. NVIDIA's [C-FoundationStereo-S card](https://huggingface.co/nvidia/c-foundationstereo-s) says downloading accepts its [license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The pinned fixed320×736v2 artifact has been acquired and inspected, and one bounded local CPU experiment completed with real raw disparity. Its [contract and execution adapter](scripts/sponsors/README.md) use version-specific NVIDIA preprocessing. The application can now consume a source-bound cached estimate for rendering alongside the unchanged measured assets. Local model execution and application consumption have been verified for one development pair, and one Nebius AI Cloud run is described below; physical accuracy remains open.

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

The separate [California offline tools](scripts/california/README.md) prepare
native AV2 observations and assemble indexed source-measurement scenes without
inventing dense depth. This bridge is verified on development input; native
ground placement, public catalog/browser admission and physical validation
remain incomplete. It does not replace the existing measured-scene workflow.

The official [DrivingStereo dataset](https://drivingstereo-dataset.github.io/) and its MIT license provide the source. The downloaded 500 matched image/depth triplets, calibrations, acquisition hashes and inspection records are in `.local/data/drivingstereo/` and are ignored by Git. The prepared redistributed subset retains its license in `public/scenes/LICENSE`.

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
  --output .local/evaluation/reprepared-scenes \
  --road-tolerance 0.08 --max-depth 80 --mesh-edge-limit 0.4
```

The pinned preparation stack was installed in an isolated environment and replayed on Python 3.13.5/macOS arm64 across five frozen development frames. Four accepted scenes reproduced all eight non-metadata artifacts byte for byte, with identical camera, transform, processing and source metadata; only the recorded processing-code identities changed. The fifth retained its road-support rejection and left the catalog unchanged. This verifies the current preparer against prior prepared outputs, without establishing physical accuracy or other-platform equivalence.

Choose a new output directory for a replay; an existing scene is retained rather than overwritten. The current adapter accepts finite horizontal rectified matrices with positive focal lengths, a left-origin projection, positive baseline and matching source sizes/fx/fy/cy. Unsupported projection terms and malformed size fields stop processing. This checks the supported matrix form, not physical calibration or exact image alignment.

Preparation reads each input file once and uses those captured bytes for decoding, calibration, source hashes and copied images/license. A later edit to a source path cannot substitute different bytes into the prepared output. Keep a stable matched acquisition during input capture: these separate file reads do not establish synchronization across files or sensors.

It creates calibrated points, supported triangles, a depth buffer, support mask, source/code hashes and per-scene license. It stages the complete scene, catalog and any new shared license before committing the catalog with an atomic local replacement. Handled errors before that commit roll back this invocation's additions; an interruption immediately after a successful commit retains the complete scene. This assumes one writer and private staging paths. Cleanup failures, concurrent writers, process death and power-loss recovery are unproven. A nonzero exit can occur after a successful commit, so inspect the catalog and scene before retrying; existing scenes are retained rather than overwritten. The frontend reads the generated catalog rather than selecting a hardcoded frame. Tests contain explicitly isolated annotations on actual source pixels; they do not supply runtime defaults.

Preparation records the two processing source files observed after imports and rechecks them immediately before publication; detected changes or unreadable files stop the job. These hashes do not authenticate already loaded bytecode, dependencies or the complete runtime. Reproduce with fixed source files and the pinned preparation environment; changes restored between observations are not detected.

```sh
.local/venv-preparation/bin/python -B -m unittest discover -s tests -p 'test_scene_preparation.py' -v
```

The Python checks exercise unsupported/malformed calibration and processing controls, preserve existing destinations, and reproduce the original geometry buffers from the real acquisition. A controlled mid-run mutation of isolated real-source copies also verifies that prepared hashes, calibration and copied files retain the captured input identity. Publication checks inject eight I/O failures and five post-operation Python interrupts, verifying rollback or retained committed output as appropriate. Their prior-scene fixture is prepared from genuine acquired raw data. If that optional raw acquisition is absent, the suite explicitly skips its real-data portion and still runs analytic/control checks.

Local hackathon rules and design QA remain ignored. Public runtime instructions and third-party notices remain with the source.

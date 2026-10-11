# Sightline

A local, single-workspace parking visibility comparison using real DrivingStereo images, sparse reference-depth data and an explicitly estimated NVIDIA reconstruction. The runtime contains no procedural street, preset actor coordinates, built-in actor dimension tables, fabricated unknown volume, or demo response. Dimensions suggested by the optional NVIDIA Nemotron assistant appear as declared, editable what-if values. Missing assets and invalid inputs stop the comparison explicitly.

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

## NVIDIA Nemotron assistant (Nebius Token Factory)

After you place the crossing, vehicle and person, the assistant row under the header accepts a plain-English what-if, such as "box truck 10 ft back, child". NVIDIA Nemotron 3.5 Lightning, called at runtime through the Nebius Token Factory chat API with tool calling, declares the vehicle size, person height and distance. Sightline applies them to the normal fields and computes Blocked or Unknown with its own geometry. Nemotron then writes a short answer from those computed results.

Check distances asks Nemotron to pick your vehicle plus up to three common parked vehicle types. Sightline tests each one locally at every 0.25 m of setback and shows where the declared vehicle stops blocking the line from the driver eye to the person's height point. The 15 ft and 20 ft marks are shown as reference. California Vehicle Code §22500(n) uses 20 ft for parking near crosswalks. Sightline does not decide legal compliance. Click a row to apply that vehicle and distance.

Nemotron never chooses image points or coordinates and never decides visibility. Every value it declares stays visible and editable. Answers that use unsupported wording or numbers absent from the computed results are replaced by the app's own computed line. Dragging, sliders and comparisons never call the model.

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

Blocked currently means the eye-to-target segment intersects the complete declared vehicle cuboid. A contact with a reconstructed reference triangle remains Unknown: adjacent sparse returns constrain its vertices, but do not establish an opaque continuous surface between them. Triangle contact, including coplanar/grazing cases, remains tested separately from this evidence policy. The calculation uses floating-point determinant and plane tolerances rather than exact arithmetic; rounding constructed near-coplanar points can affect contact. This numerical tolerance is not a sensor uncertainty model. Vehicle orientation must be a horizontal unit axis. Camera transforms must be proper orthonormal rotations and agree with the renderer's pose; invalid calibration is rejected without repair.

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

Install its separate Python dependencies from `scripts/requirements-model-inspection.txt` into a development virtual environment, then run `python -m unittest discover -s tests -p test_onnx_inspection.py -v` with that interpreter. Invocation requires a local model path, `--expected-sha256`, `--max-bytes` and `--output`; the output directory must already exist. Use the published hash and exact file size for the chosen revision. This tool does not download models or authorize their use. NVIDIA's [C-FoundationStereo-S card](https://huggingface.co/nvidia/c-foundationstereo-s) says downloading accepts its [license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The pinned fixed320×736v2 artifact has been acquired and inspected, and one bounded local CPU experiment completed with real raw disparity. Its [contract and execution adapter](scripts/sponsors/README.md) use version-specific NVIDIA preprocessing. The application can now consume a source-bound cached estimate for rendering alongside the unchanged measured assets. Local model execution and application consumption have been verified for one development pair; Nebius integration and physical accuracy remain open.

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

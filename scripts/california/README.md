# California offline data tools

## Palo Alto scene build (current)

This is the build order behind umbra's current local Palo Alto scenes, as of
2026-10-16. The app accepts `ground.json` schema 3 from placement protocol v3.4
(with a shape-v2 car shape when the suggested vehicle is annotated), and schema
4 from `measured_ground.py` (`measured-ground-v1` or `measured-ground-v2`).
Ground schemas 1 and 2 and shape-v1 are refused. Changing a schema number does
not convert an old scene.

**A fresh clone cannot yet complete this recipe unaided.** The frozen builder
protocol (builderVersion 4.3), placement protocol (revision 3.4), frozen site
split and its hash record, shape-v2 method and approach-half split addendum,
and their freeze records exist only in the maintainer's ignored local folder.
The current builds also depend on saved download listings and receipts. Obtain
a reviewed method/input bundle and matching source snapshots from the
maintainer first. Some split and lock lookups are hardcoded to ignored paths,
with no command-line override; moving those files beside the commands below
is insufficient. The full-recording support downloader and catalog installer
are also local-only. The commands below document the interfaces and order;
they are not a verified clean-clone run.

### 1. Set up Python and obtain one recording

Base-build receipts record CPython **3.13.5**, also recorded in the preparation
virtual environment. Use that version and the pinned requirements; other
Python versions/platforms are not established by these receipts. Node.js must
also be on `PATH`: measured-ground finalization runs the app's JavaScript
geometry. Follow the [app setup](../../README.md) for Node dependencies.

Run the setup from a repository root. Replace CAPS placeholders with your own
paths or values. Use full paths for input/output placeholders if switching
between source snapshots. Every `*_SHA256` is the lowercase, 64-character
SHA-256 of the exact file bytes, not a filename or a hash of parsed JSON.

```sh
python3.13 -m venv BUILD_VENV
. BUILD_VENV/bin/activate
python --version
python -m pip install -r scripts/california/requirements-av2.txt
shasum -a 256 INPUT_FILE
```

Choose a development site from the frozen split. It supplies the full log UUID,
mapped crossing ID and publisher partition (`train` or `val`); do not guess the
partition from the city. The measured-ground script currently allows only five
specific development logs, not an arbitrary Palo Alto recording. Historical
test roles remain enforced by the base builders; replaying those scenes needs
the separate frozen lock procedure. Reserve/excluded recordings are outside
this recipe.

Download that log from the public AV2 **Sensor** dataset, under
`s3://argoverse/datasets/av2/sensor/AV2_SPLIT/LOG_UUID/`. Keep publisher filenames
and this layout:

```text
BASE_BUILD_DIRECTORY/data/LOG_UUID/
  annotations.feather
  city_SE3_egovehicle.feather
  calibration/intrinsics.feather
  calibration/egovehicle_SE3_sensor.feather
  map/log_map_archive_LOG_UUID____PAO_city_MAP_NUMBER.json
  map/LOG_UUID_ground_height_surface____PAO.npy
  map/LOG_UUID___img_Sim2_city.json
  sensors/cameras/stereo_front_left/CAMERA_TIMESTAMP_NS.jpg
  sensors/cameras/stereo_front_right/CAMERA_TIMESTAMP_NS.jpg
  sensors/lidar/LIDAR_TIMESTAMP_NS.feather
```

For example, with the AWS CLI installed, this downloads the needed metadata,
both stereo cameras and LiDAR for just the selected log. It does not create the
support receipts required later. Save the publisher license/attribution notice
as `LICENSE_FILE` as well; the preparation hashes and copies that notice.

```sh
AV2_SITE=BASE_BUILD_DIRECTORY
AV2_LOG_ID=LOG_UUID
AV2_PARTITION=AV2_SPLIT
AV2_SCENE_ID=SCENE_ID
mkdir -p "$AV2_SITE/data/$AV2_LOG_ID" "$AV2_SITE/listings"
aws s3 cp --no-sign-request --recursive \
  "s3://argoverse/datasets/av2/sensor/$AV2_PARTITION/$AV2_LOG_ID/" \
  "$AV2_SITE/data/$AV2_LOG_ID/" \
  --exclude '*' --include 'annotations.feather' \
  --include 'city_SE3_egovehicle.feather' --include 'calibration/*' \
  --include 'map/*' --include 'sensors/cameras/stereo_front_left/*' \
  --include 'sensors/cameras/stereo_front_right/*' --include 'sensors/lidar/*'
for AV2_CAMERA in stereo_front_left stereo_front_right; do
  curl --fail --location \
    "https://argoverse.s3.amazonaws.com/?list-type=2&prefix=datasets/av2/sensor/$AV2_PARTITION/$AV2_LOG_ID/sensors/cameras/$AV2_CAMERA/&delimiter=/&max-keys=1000" \
    --output "$AV2_SITE/listings/$AV2_CAMERA.xml"
done
curl --fail --location \
  "https://argoverse.s3.amazonaws.com/?list-type=2&prefix=datasets/av2/sensor/$AV2_PARTITION/$AV2_LOG_ID/sensors/lidar/&delimiter=/&max-keys=1000" \
  --output "$AV2_SITE/listings/lidar.xml"
```

The three saved XML listings must be complete (`IsTruncated` is `false`) and
belong to these sensors in this log. Preserve their exact bytes for replay.
The builder uses their timestamps, sizes and ETags to select and check sources.
It can choose with metadata first, but writes `plan.json` only when the chosen
stereo pair and frame sweep are present and match the listings.

### 2. Select, prepare and assemble the base scene

Use the source snapshot whose runtime hashes match the supplied builder
protocol. The recorded builds used builderVersion 4.3, variant `v4-18.5`;
today's builder also contains version 5 and will reject the old protocol's code
pins. The two historical 4.3 protocol bundles also pin different versions of
the shape helper. Match the whole bundle to its receipt, not just the builder
version number. Do not regenerate a protocol merely to bypass a mismatch.

From that build source root, with the environment from step 1 active:

```sh
python -I -B scripts/california/build_av2_scene.py \
  --protocol BUILDER_PROTOCOL_FILE --protocol-sha256 BUILDER_PROTOCOL_SHA256 \
  --placement-protocol PLACEMENT_TEMPLATE_FILE \
  --placement-protocol-sha256 PLACEMENT_TEMPLATE_SHA256 \
  --split SPLIT_FILE --split-lock SPLIT_HASH_RECORD_FILE \
  --log-id "$AV2_LOG_ID" --crossing-id CROSSING_ID \
  --log-dir "$AV2_SITE/data/$AV2_LOG_ID" \
  --listing-left "$AV2_SITE/listings/stereo_front_left.xml" \
  --listing-right "$AV2_SITE/listings/stereo_front_right.xml" \
  --listing-lidar "$AV2_SITE/listings/lidar.xml" \
  --license LICENSE_FILE --protocol-variant v4-18.5 \
  --output "$AV2_SITE/builder"
```

Continue only if `builder/selection.json` says `plan-written`. Use its
`protocolSceneValues.scene.sceneId` for `AV2_SCENE_ID`. A missing source or
infeasible frame is a recorded failure, not permission to invent a crop or
placement. Each tool requires a new output directory. If a metadata-only first
pass reports missing sensors, download the exact files it names and use a new
builder output directory for the second pass.

Then run:

```sh
python -I -B scripts/california/prepare_av2.py \
  --plan "$AV2_SITE/builder/plan.json" --output "$AV2_SITE/prepared"
python -I -B scripts/california/assemble_av2_scene.py \
  --manifest "$AV2_SITE/prepared/manifest.json" \
  --output "$AV2_SITE/assembled" --id "$AV2_SCENE_ID"
```

Preparation reads all ten sources named in `plan.json` and writes
`manifest.json`, `left.png`, `right.png`, `calibration.json`, `references.npz`
and `LICENSE-AND-ATTRIBUTION.md`. Assembly reads those outputs and writes the
indexed scene, recorded image, binary buffers and an initial catalog entry.
Keep the original source/prepared directories available for the ground step.

### 3. Freeze the per-scene placement inputs; measure paint if needed

The template's example scene values do not describe every log. Before grounding,
obtain a derived, hash-recorded `SCENE_PLACEMENT_PROTOCOL_FILE` (revision 3.4).
Its `scene` identity, timestamps, crop, lane ID in `mapElements.roadAxis` and
`currentVehicle.trackPrefix` must come from the builder's
`protocolSceneValues`. For a declared vehicle the prefix is `declared-default`,
not null. Preserve the template's method rules and the recorded fixed-versus-fit
height-band choice. The tracked evaluation runner's `scene_protocol` method in
`scripts/evaluation/m7/run_heldout.py` performs this derivation as part of a
larger run; the builder has no standalone command for it. The reviewed
per-scene protocol and its freeze record are currently missing public inputs.

Paint measurement is separate. The current template already contains earlier
paint-v4 evidence for two development crossings; other crossings can remain
`unmeasured`. To reproduce that evidence, obtain the frozen paint protocol
(version 4), its matching archived `measure_paint_edge.py`, a config and sweep
download receipts. The current script supports later method work too; its bytes
are not the archived paint-v4 code. Use the code named by the paint receipt:

```sh
python -I -B PAINT_CODE_FILE select \
  --protocol PAINT_PROTOCOL_FILE --protocol-sha256 PAINT_PROTOCOL_SHA256 \
  --config PAINT_CONFIG_FILE --output PAINT_SELECTION_FILE
python -I -B PAINT_CODE_FILE measure \
  --protocol PAINT_PROTOCOL_FILE --protocol-sha256 PAINT_PROTOCOL_SHA256 \
  --config PAINT_CONFIG_FILE --sweeps PAINT_SWEEP_RECEIPTS_FILE \
  --output PAINT_RESULT_FILE
```

The config needs `schemaVersion: 1`, `logId`, `crossingId`, and `sources` records
for `poses`, `extrinsics`, `map`, `groundRaster`, `groundTransform` and
`lidarListing`, each with `path` and `sha256` (record byte counts too). Relative
paths resolve beside the config. `select` reads metadata only; download its
chosen sweeps before `measure`. The sweep receipt uses a `files` array with
`timestampNs`, `path` and `sha256` for every selected sweep; paths resolve beside
that receipt. Paint-v4 selects 15 sweeps. Its downloader/receipt writer is
local-only, not an extra subcommand of `measure_paint_edge.py`.

Neither paint command updates ground, suggestions or the catalog. A measured
label must be carried in the per-scene protocol's `crossingError.perCrossing`
entry with its evidence file/hash; unresolved evidence stays unresolved. Do
this before freezing the per-scene protocol and building ground/suggestions.
The mapped crossing is never moved by paint measurement. Later evaluation
paint receipts alone do not attach their result to an app scene.

### 4. Build ground schema 3 and the car shape, then initial suggestions

For a builder selection with `carShape.applies: true`, first build a temporary
box-only ground layer. It supplies the averaged annotated pose to the shape
planner; do not generate suggestions from this intermediate layer.

```sh
python -I -B scripts/california/prepare_av2_ground.py \
  --scene "$AV2_SITE/assembled/scene.json" --scene-sha256 ASSEMBLED_SCENE_SHA256 \
  --prepared "$AV2_SITE/prepared/manifest.json" --prepared-sha256 MANIFEST_SHA256 \
  --protocol SCENE_PLACEMENT_PROTOCOL_FILE --protocol-sha256 SCENE_PROTOCOL_SHA256 \
  --annotations "$AV2_SITE/data/$AV2_LOG_ID/annotations.feather" \
  --annotations-sha256 ANNOTATIONS_SHA256 \
  --output "$AV2_SITE/grounded-box" --no-car-shape
python -I -B scripts/california/car_shape.py plan \
  --log "$AV2_LOG_ID" \
  --ground "$AV2_SITE/grounded-box/ground.json" --ground-sha256 BOX_GROUND_SHA256 \
  --annotations "$AV2_SITE/data/$AV2_LOG_ID/annotations.feather" \
  --annotations-sha256 ANNOTATIONS_SHA256 \
  --poses "$AV2_SITE/data/$AV2_LOG_ID/city_SE3_egovehicle.feather" \
  --poses-sha256 POSES_SHA256 \
  --listing-file "$AV2_SITE/listings/lidar.xml" --listing-sha256 LIDAR_LISTING_SHA256 \
  --out-root "$AV2_SITE/shape"
python -I -B scripts/california/car_shape.py download \
  --log "$AV2_LOG_ID" --plan-sha256 SHAPE_PLAN_SHA256 --out-root "$AV2_SITE/shape"
python -I -B scripts/california/prepare_av2_ground.py \
  --scene "$AV2_SITE/assembled/scene.json" --scene-sha256 ASSEMBLED_SCENE_SHA256 \
  --prepared "$AV2_SITE/prepared/manifest.json" --prepared-sha256 MANIFEST_SHA256 \
  --protocol SCENE_PLACEMENT_PROTOCOL_FILE --protocol-sha256 SCENE_PROTOCOL_SHA256 \
  --annotations "$AV2_SITE/data/$AV2_LOG_ID/annotations.feather" \
  --annotations-sha256 ANNOTATIONS_SHA256 --output "$AV2_SITE/grounded" \
  --car-shape-dir "$AV2_SITE/shape/shape-v2-$AV2_LOG_ID" \
  --car-shape-plan-sha256 SHAPE_PLAN_SHA256
SOURCE_DATE_EPOCH=UNIX_SECONDS python -I -B scripts/california/suggest_av2_placements.py \
  --grounded "$AV2_SITE/grounded" --metadata-sha256 GROUNDED_SCENE_SHA256 \
  --buffers "$AV2_SITE/assembled" \
  --protocol SCENE_PLACEMENT_PROTOCOL_FILE --protocol-sha256 SCENE_PROTOCOL_SHA256
```

`plan` writes `shape/shape-v2-LOG_UUID/sweep-plan.json` and `listing.xml`, and
prints the plan hash. Record that hash before `download`, which writes
`receipt.json` and `sweeps/TIMESTAMP_NS.feather` beside the plan. The final ground
build verifies those receipts and embeds `carShape` in `ground.json`; there is
no separate runtime car-shape file. Unsupported shape cells remain null.

When `carShape.applies` is false, skip the temporary ground and shape commands.
Run the final ground command with `--no-car-shape` instead of both
`--car-shape-*` flags, then generate suggestions. That scene has `carShape:
null`; a shape is required exactly when suggestions name an annotated
stationary vehicle. A real AV2 vehicle is described only as stationary.

Ground reads the assembled metadata/reference buffers, prepared manifest and
references, publisher map/raster/transform/poses, annotations and per-scene
protocol. It writes `scene.json`, `ground.json` and `terrain.bin` in `grounded`;
the buffers remain in `assembled`. The suggestion step reads these plus
`referencepositions.bin` and `referenceeligible.bin`, and writes
`suggestions.json` and `catalog-entry.json`. Use a fixed integer
`SOURCE_DATE_EPOCH` to reproduce its timestamp. Neither file can already exist.

### 5. Build measured-ground-v1, then measured-ground-v2

Run these commands using the **current app source**, which contains
`measured_ground.py` and its matching JavaScript geometry. `AV2_SITE` still
names the original base build, with `assembled/`, `grounded/` (including
suggestions and catalog entry) and `data/LOG_UUID/` intact.

V1 needs a receipted set of other sweeps. For annotated-vehicle builds this is
normally `AV2_SITE/shape/shape-v2-LOG_UUID/receipt.json`, together with its
`sweep-plan.json` and `sweeps/` directory. The declared-vehicle builds used
separate near-time support receipts/listings/sweeps. Pass the receipt explicitly
to avoid an ignored-path fallback. V1 uses eligible sweeps within one second of
the frame, excludes the frame and the inclusive ±0.25 second neighbourhood,
and holds every fourth eligible sweep back for calibration.

V2 requires a **full-recording support bundle**, not the car-shape receipt:
`receipt.json`, its hash-pinned `listing.json`, and `sweeps/TIMESTAMP_NS.feather`.
The recorded full bundles used a 30-second window around the frame and omitted
the frame sweep. There is currently no tracked command that produces this
bundle: obtain the receipted bundle or a reviewed downloader from the maintainer.
Raw LiDAR files alone are insufficient. `receipt.json` must have
`schema: "support-sweeps-download-receipt"`, `log`, `split`,
`frameSweepTimestampNs`, `listingSha256`, `rule.excludesFrameSweep: true`, and a
`sweeps` array. Each sweep records `timestampNs`, `key`, `file`, `status`,
`bytes`, `contentLength`, `etag`, `md5`, `md5MatchesEtag`, `sizeMatches` and
`sha256`. The script checks these against the saved listing and actual bytes.

```sh
python -B scripts/california/measured_ground.py \
  --site "$AV2_SITE" --out MEASURED_V1_DIRECTORY \
  --version measured-ground-v1 --band-range-rule bins \
  --support-receipt NEAR_SUPPORT_RECEIPT_FILE
python -B scripts/california/measured_ground.py \
  --site "$AV2_SITE" --out MEASURED_V2_DIRECTORY \
  --version measured-ground-v2 --band-range-rule bins \
  --tier-0 MEASURED_V1_DIRECTORY --support-receipt FULL_SUPPORT_RECEIPT_FILE
```

Both commands need new output directories. They copy `assembled/` and
`grounded/`, write schema-4 ground and a `measured-ground-protocol.json`, and
refresh suggestions and catalog hashes themselves. **Do not rerun
`suggest_av2_placements.py` on schema 4:** that script accepts the earlier
schema-3 input for v3.4, not a measured-ground layer.

V2 keeps every v1 height and its calibration exactly as tier 0, fills only
supported v1 holes using additional same-recording sweeps, and writes the exact
v1 ground bytes to `MEASURED_V2_DIRECTORY/tier-0-ground.json`. Retain the whole
v1 directory: v2 also reads its catalog ground pin, `calibration-points.json`
and `calibration-errors.json`. Missing ground or calibration remains Unknown.
These are development builds, not a new held-out accuracy pass.

### 6. Put the finished files in the app and update its catalog

The app fetches `/scenes/index.json`; it does not discover build folders.
Install into `public/scenes/SCENE_ID/` under the current app source. The local
installer copies `assembled` first, then overlays the final `grounded` files,
excluding both intermediate `catalog-entry.json` files. For v2 it then copies
`tier-0-ground.json` from the build root. This preserves the final metadata
instead of replacing it with the ungrounded assembled metadata.

| Files in `public/scenes/SCENE_ID/` | Source |
| --- | --- |
| `positions.bin`, `colors.bin`, `triangles.bin` (including the empty triangle file) | Final build's `assembled/` |
| `sourceindices.bin`, `referencepositions.bin`, `referencecamerapositions.bin`, `referenceuv.bin`, `referenceeligible.bin` | Final build's `assembled/` |
| `recorded.png`, `LICENSE-AND-ATTRIBUTION.md` | Final build's `assembled/` |
| `scene.json`, `ground.json`, `terrain.bin`, `suggestions.json` | Final build's `grounded/` |
| `measured-ground-protocol.json` | Final measured build's `grounded/`; keep for provenance |
| `tier-0-ground.json` | V2 build root only; required by the loader for v2 |

For a new scene directory, these are the manual copy commands. Set
`AV2_FINAL` to the v2 output, or to the v1/base build if installing that version;
omit the last copy unless it is v2. No tracked catalog-install tool currently
performs these steps.

```sh
AV2_FINAL=FINAL_BUILD_DIRECTORY
mkdir -p public/scenes
mkdir "public/scenes/$AV2_SCENE_ID"
cp "$AV2_FINAL/assembled/"*.bin "$AV2_FINAL/assembled/recorded.png" \
  "$AV2_FINAL/assembled/LICENSE-AND-ATTRIBUTION.md" "public/scenes/$AV2_SCENE_ID/"
cp "$AV2_FINAL/grounded/scene.json" "$AV2_FINAL/grounded/ground.json" \
  "$AV2_FINAL/grounded/terrain.bin" "$AV2_FINAL/grounded/suggestions.json" \
  "public/scenes/$AV2_SCENE_ID/"
cp "$AV2_FINAL/grounded/measured-ground-protocol.json" "public/scenes/$AV2_SCENE_ID/"
cp "$AV2_FINAL/tier-0-ground.json" "public/scenes/$AV2_SCENE_ID/"
```

Omit the protocol copy too for a base schema-3 build. Copy the object from the
**final** `grounded/catalog-entry.json` into the JSON array in
`public/scenes/index.json`, replacing the same `id` if present and preserving
other entries. Create the file as a JSON array if it is absent. A catalog
object has these fields (one example ID; replace the
hash placeholders with the final generated values):

```json
{
  "id": "av2-pao-c780d53a-2069452",
  "label": "Palo Alto 1 · AV2 c780d53a (development)",
  "metadataSha256": "FINAL_SCENE_SHA256",
  "ground": {"file": "ground.json", "sha256": "FINAL_GROUND_SHA256"},
  "suggestions": {"file": "suggestions.json", "sha256": "FINAL_SUGGESTIONS_SHA256"}
}
```

`id` must match the directory and `scene.json` and contain only letters, digits,
`_` or `-`. `label` is display text; the current menu uses its part before
` · `. `metadataSha256` pins the raw final `scene.json` bytes. The catalog
`ground` pin must equal `scene.measurements.ground`'s file/hash; that scene
descriptor also records its exact byte length. `suggestions` pins the final
suggestion bytes. Both catalog descriptors have exactly `file` and `sha256`.

The loader additionally checks `scene.artifactHashes` for the render buffers,
`scene.measurements.artifactHashes` for reference buffers, the recorded PNG's
hash/length/dimensions, and `ground.terrain`'s hash/length. V2 also pins
`tier-0-ground.json` by `ground.measuredGround.tier0.groundSha256` and checks
that tier 0 exactly preserves v1. Suggestions must name the same hashes for
`scene.json`, `ground.json`, `terrain.bin`, `recorded.png`,
`referencepositions.bin`, `referenceeligible.bin` and `protocol.json` (the
ground layer's protocol hash). A stale file/pin fails loading; copying only
`scene.json` or only editing the catalog cannot install a scene. Preserve the
source snapshots, protocols, listings and receipts separately for provenance;
the browser does not fetch those original sources to fill missing evidence.

## Native AV2 stereo preparation

`prepare_av2.py` prepares one explicitly selected native Argoverse 2 Sensor
observation from local publisher snapshots. It reads exact camera/sweep poses,
rectifies the stereo images, checks actual source support and retains every
original LiDAR return separately. It does not download data, accept terms,
execute a model, change the scene catalog or admit a location into the app.

```sh
python3 scripts/california/prepare_av2.py \
  --plan /explicit/av2-plan.json --output /explicit/new-preparation
python3 -m unittest discover -s tests/california -p test_prepare_av2.py -v
```

Use the versions in `requirements-av2.txt` in an isolated environment. The
verified environment is Python 3.13.5 on macOS arm64; other platforms need their
own verification. This tool is separate from the narrower DrivingStereo
preparation environment.

The plan declares `schemaVersion: 1`, `sourceClass: "Argoverse2Sensor"`,
`logId`, decimal-string `cameraTimestampNs`, `lidarTimestampNs` and `crossingId`,
`rectifiedSize: [width,height]`, and `crop: [x,y,width,height]`. Its `sources`
object requires `left`, `right`, `lidar`, `poses`, `intrinsics`, `extrinsics`,
`map`, `groundRaster`, `groundTransform` and `license`. Each source record has
an explicit `path`, `bytes` and `sha256`; relative paths resolve from the plan
directory. Preserve publisher filenames/sensor directories. Source hashes
describe captured local bytes and do not establish authorization or remote
authenticity. Verify applicable terms, attribution and intended use separately.

The output crop manifest, PNG images and calibration are compatible with the
existing NVIDIA input reader, with AV2 rectification and crop transforms
recorded explicitly. Reader compatibility does not prove inference, physical
calibration or sponsor eligibility. `references.npz` preserves source indices,
floating coordinates, projection/support masks and timing; it is not an
image-sized depth buffer. No interpolation or collision overwrite manufactures
missing camera-depth evidence.

New preparations also include `processing.publisherReportedTiming`. It derives
return ages from the dataset-reported sweep start plus each `offset_ns`, relative
to the camera timestamp. Full, left-eligible, right-eligible and both-eligible
cohorts retain complete counts and min/max/linear median/p95. Temporal values
are exact decimal nanosecond strings, including fractional percentiles; empty
cohorts have zero counts and null temporal statistics. The original plan
timestamp strings and `lidarMinusCameraNs` remain intact.

These are publisher-reported timing diagnostics. Clock/exposure precision,
dynamic-object motion, camera visibility and physical calibration remain
unvalidated. Source coordinates already have ego-motion compensation; this
summary preserves them and the existing geometric eligibility masks. Earlier
prepared/model/cache records remain immutable. See the pinned publisher
[Sweep contract](https://github.com/argoverse/av2-api/blob/b7321d1f71f6ce0ecdd151f4f2b648338c191edd/src/av2/structures/sweep.py).

Preparation may observe only part of the nominal mapped crossing. All four
corners remain in `processing.mappedCrossing`, with camera XYZ, crop UV and a
per-view `cropSupported` boolean for each corner. Behind-camera or out-of-crop
corners remain unsupported; undefined projection coordinates are `null`.
Every pixel in the selected RGB crop must still have valid source support in
both cameras. These diagnostics do not establish whole-crossing coverage,
ground contact, physical accuracy or runtime admission.

Native publisher map JSON can contain `NaN` in unused non-crossing geometry.
Those values remain explicit source unknowns; required crossing corners must
still be finite. Duplicate keys, infinities and overflow are rejected. This
map-specific decoding policy does not relax plan or ground-transform parsing.

Input limits are 16 MB per source, 64 MB total, 16 million decoded/rectified
pixels, dimensions at most 32,766, and eight million float16 ground samples.
These are operating limits, not measured memory or accuracy guarantees.

Publisher terrain is separately labeled nominal cityZ ground. NaNs remain
missing, and negative continuous raster coordinates that would alias index
zero through truncation are rejected. Terrain cannot establish camera-visible
depth, free space, confidence or Clear. Exact poses account for ego movement;
dynamic objects, asynchronous capture, occlusion and physical uncertainty
remain unresolved. Prepared outputs are development artifacts; runtime
integration, independent physical references and location admission require
additional verification. Output creation refuses existing destinations.

## Indexed scene assembly

`assemble_av2_scene.py` converts an authenticated native preparation into a
new scene directory consumed by the application's indexed-measurement loader.
It requires the existing 736×320 crop contract. Use the same isolated AV2
environment; no further dependencies or model execution are required.

```sh
python3 scripts/california/assemble_av2_scene.py \
  --manifest /explicit/preparation/manifest.json \
  --output /explicit/new-indexed-scene --id explicit-observation-id
python3 -m unittest discover -s tests/california -p test_assemble_av2_scene.py -v
```

Every source row retains its ID, Float64 world/camera XYZ, floating crop UV and
eligibility. Render points are a separate Float32 view of every left-eligible
row, with projected image color and zero triangles. The world origin is the
actual left camera, with city altitude mapped to world Y; Y=0 is camera altitude,
not a road plane. The camera uses the virtual rectified R1 pose and cropped P1.
The original reference NPZ and source identities remain bound in provenance.

The loader rejects absent or colliding pixel ownership; it never fabricates a
depth raster, borrows a neighbor or promotes nominal terrain to observed road.
Native ground placement and movement remain unvalidated. Declared cuboid
contact can establish geometric Blocked; sparse free space remains Unknown.
This tool does not add assets to the public scene catalog, attach a prediction,
or establish physical accuracy, camera visibility or California site admission.

The assembler bounds manifest bytes, prepared bytes, ZIP expansion and NPY
headers before materializing arrays, rejects infinities, preserves unavailable
NaN rows, and refuses an existing destination. Artifact checksums and metadata
syntax do not authenticate self-declared provenance or image correspondence;
source-bound review of output identity/color/calibration is separately required.
The assembler emits `recordedImage` metadata for its exact PNG and a separate
`catalog-entry.json` containing the raw scene-metadata SHA256. Native loading
requires a trusted caller to pass that digest as the third argument to
`loadScene(fetchResource, id, metadataSha256)`. The loader captures bounded
image bytes, verifies their hash and PNG header dimensions, and supplies a Blob;
the app checks actual decoded dimensions and revokes owned image URLs.
Native input does not request a dense support mask or display a depth-coverage
percentage. Caller checksums establish artifact consistency, not authentication
of an arbitrary external catalog or proof of physical camera alignment.
Catalog admission, valid ground placement and independently referenced physical
accuracy remain unestablished. This tool does not modify the public catalog.

## Native cached NVIDIA rendering

`attach_av2_prediction.py` attaches an existing, source-bound NVIDIA prediction
to an indexed native scene as renderer-only points with zero triangles. It
validates the original cache, prepared stereo pixels/tensors/calibration and
current measured provenance separately. It does not run inference. Use the
existing inspection environment that produced the cache; environment drift
fails validation.

```sh
python3 scripts/california/attach_av2_prediction.py \
  --scene /explicit/measured/scene.json --scene-sha256 EXPECTED_SHA256 \
  --prepared /explicit/preparation/manifest.json --prepared-sha256 EXPECTED_SHA256 \
  --producer /explicit/original/prediction.json --producer-sha256 EXPECTED_SHA256 \
  --plan /explicit/original/plan.json --plan-sha256 EXPECTED_SHA256 \
  --input-equivalence /explicit/review/result.json --input-equivalence-sha256 EXPECTED_SHA256 \
  --model /explicit/original/model.onnx --output /explicit/new-attached-scene
```

For a prediction made directly from the exact current native preparation,
replace both `--input-equivalence` arguments with `--direct-native`. This mode
requires the original model-input manifest path and bytes to match `--prepared`,
and the original observation alias to match the measured scene ID. It retains
all original source, tensor, calibration, pose, reference and cache checks.
The resulting `input-binding.json` records direct identity; it is not a
historical equivalence proof. The two modes are exclusive and never selected
automatically. Do not relabel a cached output or rerun inference for a label.

The model path resolves the historical execution record; this conversion does
not load or rehash weights. Matching metadata is not independent attestation
that an asserted inference occurred; retain original execution evidence and
independently audit the actual output. Caller hashes must come from trusted
source capture. The known historical inspection descriptor has a hash without
a byte count; its capture still enforces the same finite byte ceilings.

The predicted display uses the scene's existing axial camera clips and the
right-input pixel domain. These are rendering/input limits, not confidence or
accuracy filters. Full raw disparity, validity and converted depth, including
NaNs and extreme tails, remain separately hash-bound. Measured buffers,
ground policy and query ownership stay separate; absent support remains
Unknown. The tool emits a new caller metadata digest and refuses an existing
output. It does not modify the public catalog or establish ground, physical
accuracy, a usable California site or complete sponsor eligibility.

## Selective probe inspection

`inspect_probe.py` is a standard-library, offline check of an explicit AV2
acquisition plan and saved publisher S3 listing XML. It does not download data,
accept terms, derive scene geometry or admit a usable location.

```sh
python3 scripts/california/inspect_probe.py \
  --plan /explicit/plan.json \
  --listing /explicit/root-listing.xml \
  --listing /explicit/calibration-listing.xml \
  --listing /explicit/map-listing.xml \
  --listing /explicit/left-listing.xml \
  --listing /explicit/right-listing.xml \
  --listing /explicit/lidar-listing.xml \
  --output /explicit/new-observations.json
python3 -m unittest discover -s tests/california -v
```

Missing/changed listing objects and missing approved payloads reject the inventory
and return nonzero. Pending-approval metadata-only inspection can succeed without
payloads. Successful inspection
means the inventory matched, while admission remains `unresolved`. The PAO map
tag is city evidence only. Physical site grouping stays null and split stays
unassigned. Nanosecond timestamps remain decimal strings. Every read input is
byte-pinned as an explicit external read-only reference.

`--local-root` optionally hashes size-matched payloads from an explicit AV2 log
root, bounded to a 16 MB selection. It does not read payloads while the plan's
approval flag is pending. A plan flag records approval; the caller must verify
actual authorization independently. Output cannot overwrite an input or write
inside the raw root. Output creation is exclusive and refuses any existing file,
including hard-link aliases; reruns require a new output path. Payload presence,
matching sizes and byte hashes do not
establish source authenticity, calibration accuracy or independent references.

The tests use labeled analytic byte/listing fixtures, not California samples or
runtime geometry. Real payload inspection and any later SDK conversion require
separate input/term authority and source-pinned verification.

Listing bucket/prefix checks bind the declared selection, not publisher
authenticity. Byte hashes describe each read snapshot and the observed source
file; they do not authenticate imported code or provide a cross-file atomic
snapshot. Concurrent source mutation and output write/cleanup failures are not
validated transaction guarantees.

## Map-first ground layer and placement suggestions

This section documents the earlier chain (placement protocol v3, ground schema
2), which the current app refuses. Use [Palo Alto scene build (current)](#palo-alto-scene-build-current)
for the current build and catalog installation; the older method is retained here as history.

`prepare_av2_ground.py` adds a mapped-ground layer to one assembled indexed
AV2 scene. `suggest_av2_placements.py` then writes schema-v3 placement
suggestions from it. Both read a frozen placement protocol (tolerances,
support classes, declared defaults) by explicit path and sha256. They do not
download data, run a model, change the app catalog or publish anything. Use the
same AV2 environment (`requirements-av2.txt`).

```sh
python3 -I -B scripts/california/prepare_av2_ground.py \
  --scene /explicit/assembled/scene.json --scene-sha256 SCENE_SHA256 \
  --prepared /explicit/prepared/manifest.json --prepared-sha256 MANIFEST_SHA256 \
  --protocol /explicit/protocol.json --protocol-sha256 PROTOCOL_SHA256 \
  --annotations /explicit/annotations.feather --annotations-sha256 ANNOTATIONS_SHA256 \
  --output /explicit/new-grounded
SOURCE_DATE_EPOCH=INTEGER python3 -I -B scripts/california/suggest_av2_placements.py \
  --grounded /explicit/new-grounded --metadata-sha256 GROUNDED_SCENE_SHA256 \
  --buffers /explicit/assembled --protocol /explicit/protocol.json --protocol-sha256 PROTOCOL_SHA256
python3 -I -B -m unittest discover -s tests/california -p test_prepare_av2_ground.py -v
python3 -I -B -m unittest discover -s tests/california -p test_suggest_av2_placements.py -v
```

The ground step writes a new directory with `ground.json`, `terrain.bin` and a
grounded `scene.json`. `terrain.bin` is a float32 window of the publisher
float16 ground raster (NaN where the publisher has no value), sampled with
exactly the `prepare_av2.ground_sample` rule: no interpolation or hole filling.
`ground.json` holds the mapped drivable-area outlines, crossing edges, ego lane
and annotated cuboids in scene coordinates, the tolerance
(`roadToleranceMeters`, which never goes in scene `processing`), the raster
window description and the LiDAR support rows. A support row is a
camera-eligible return, not behind an annotated cuboid, whose height agrees
with the raster; every return is tested, with no height pre-filter. The grounded
`scene.json` differs from the assembled one only in
`measurements.groundPolicy: "mappedGround"`, a hash-pinned
`measurements.ground` descriptor, a relative `originalSourceRecordBase` and
the ground wording. Its buffers stay byte-identical to the assembled scene.

With a protocol v3 file (`protocolVersion` 3) the ground step also writes
`ground.json` schemaVersion 2. Protocols v1 and v2 still give the schema-1
layer exactly as before. Version 3 adds three things:

- **Averaged stationary vehicle.** The current vehicle (the protocol's track
  prefix) is drawn at the average of all its annotations in the log. Its
  centre is the mean, its heading is the circular mean, and its size is the
  median. The box is upright. This happens only when the track passes the
  stationary check and the prefix names one track at the sweep.
  `poseAverage` records the spread and how far the sweep box was from the
  average. Every other cuboid stays at the sweep time.
- **Fitted uncertainty band.** The along-ray uncertainty uses a height band
  (`alongRayUncertainty.heightBandMeters`). It is fitted on this development
  scene's camera-visible LiDAR ground returns: the nearest-rank 95th
  percentile of their height difference from the raster, rounded up to
  1 mm. The fit uses returns inside the drivable area within 30 m and within
  0.25 m of the raster. The file also records coverage by range band, for a
  click at a return's exact projection and at its nearest pixel, and a
  split-half check. This coverage is in-sample (the band is tested on the
  returns it was fitted on), so it is not evidence that the band is adequate.
  The script refuses to fit on held-out or reserve logs of the frozen Palo
  Alto split. A held-out scene uses a fixed band fitted on development scenes
  (protocol method `fixed`): the file then records that scene's residuals and
  coverage as `heldOutResiduals`, a test of the band, never a fit, with no
  minimum count and no split-half refit.
- **Range limit.** A placement more than 30 m from the camera (horizontal) is
  refused: `Too far to place (over 30 m).` The limit is stored as
  `placementRangeLimitMeters`.

The suggestion step needs protocol v3 and a schema-2 ground layer. It refuses
protocols v1 and v2, whose outputs are historical and were written by earlier
versions of this script. Each object (crossing, vehicle front, person) gets
two positions:

- `pixelUV` is the exact floating projection of its mapped point. It is the
  position the app evaluates while the suggestion is untouched.
- `pixel` is the integer pixel that contains `pixelUV`. It is the marker and
  nudge anchor.

The object's ground point comes from the same rule the app applies: the
first hit of the `pixelUV` camera ray on the mapped ground raster, within
30 m and inside the mapped drivable area. There is no rounding, no search and
no fallback pixel. If that ray finds no valid ground, the object stays
Unknown with the reason. `pixelOffsetFromMappedM` is the remaining distance
to the mapped point. It is an internal consistency check, not accuracy, and
it is zero unless the raster blocks the point along its own ray.
`integerPixelCheck` records what the integer pixel alone would give.

Each point has a support class: `observed-nearby` (a camera-visible return
agrees within 0.5 m), `map-only` (shown as "mapped, not observed"), or
`unknown` (not placed). It also has a `hidden` flag, set when the ray passes
through an annotated cuboid first; the flag is null when annotations are not
supplied. The along-ray uncertainty is the fitted band divided by the sine of
the ray's angle below horizontal. The crossing is the mapped near edge on the
current vehicle's centre line. Its error is a separate `crossingError` field:
the mapped edge plus a ± paint offset. The status is `pending-measurement`,
`unmeasured`, `measured` (with `plusMinusM` and evidence) or `unresolved`.
Paint never moves the crossing. The road axis comes from the mapped ego lane,
and its skew is reported, never used as a limit. The curb is the
drivable-area edge, a proxy rather than a measured curb.

An annotated stationary vehicle is labeled "stationary vehicle (AV2
annotation); reason for stopping unknown". No output states why it stopped,
and the script refuses to write output containing any word built on "park".
The app draws this vehicle with the averaged annotated box from `ground.json`:
its centre, horizontal heading and dimensions. The front pixel only names that
box. The what-if vehicle is the same box moved along the lane axis, so both
start as the same box. If the size changes or a declaration replaces the vehicle while the front
pixel is still within the nudge allowance, the app uses a declared box on the
lane axis whose front is the averaged box's front point (`mappedPoint`;
protocol v3.1 `currentVehicle.declaredReplacement`). With protocol v3.1 the
script says so in `appModel` and one limitation line. The front pixel's own
ray hit is not used then: on AV2 8c the raster blocks that ray 1.03 m short of
the front. If the front pixel moves more than the nudge allowance, the app
uses a declared box on the lane axis at that pixel's ground hit.
`frontResidualToAnnotatedM` and `yawToRoadAxisDeg` record how far the vehicle
front's ground point is from the averaged front, and how far the box is
turned from the road axis. Without annotations the vehicle is a declared
default or Unknown. Without LiDAR support, points are map-only or Unknown.
Both scripts refuse to overwrite existing outputs. Output files contain
relative paths only, and identical inputs give byte-identical outputs.

Mapped ground is publisher terrain, not observed road. The LiDAR agreement
check and the band fit are consistency checks, because the publisher built
the map from LiDAR and the LiDAR shares the camera's calibration. These layers
do not establish free space, Clear, contact, why a vehicle stopped, or physical
accuracy. Results stay Blocked or Unknown. The data stays under CC BY-NC-SA
4.0 with the "(c) 2021 Argo AI, LLC" attribution recorded in both files.

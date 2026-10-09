# California offline data tools

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

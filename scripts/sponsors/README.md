# Local sponsor adapters

These tools prepare a real recorded stereo crop and an evidence-grounded Nemotron
quality audit. Preparation is not NVIDIA/Nebius execution or hackathon eligibility.
No tool downloads weights, accepts licenses, dispatches HTTP/cloud jobs, reads keys,
or edits a budget ledger. Vehicle movement and comparison remain local.

Use the existing NumPy/Pillow preparation environment. `common.publish` reuses the
exclusive-directory publisher in `scripts/prepare-stereo-crop.py`; it requires that
environment even for Token Factory artifact publication. The optional NVIDIA run
also requires ONNX and ONNX Runtime in a reviewed execution environment. Neither
is installed by these tools. Keep this execution environment separate from the
frontend and pin its package versions. CPU compatibility must be established for
the actual artifact, input and runtime; package presence is insufficient.

```sh
python -m unittest discover -s tests/sponsors -v
python scripts/sponsors/local_checks.py --crop CROP_MANIFEST --ledger SHARED_LEDGER --output NEW_PRIVATE_OUTPUT
python scripts/sponsors/nvidia_stereo.py prepare --crop CROP_MANIFEST --observation-id OBSERVATION_ID --output NEW_PLAN
python scripts/sponsors/token_factory.py prepare --evidence EVIDENCE_JSON --output NEW_PLAN
```

Outputs must be new directories. Paths in records are relative to the manifest
parent; external source references have `readOnlyExternal:true`. Existing outputs
are preserved. Code hashes identify observed source files rather than authenticated
loaded bytecode; input capture is sequential rather than a synchronized acquisition.
Atomic exclusive publication does not establish power-loss durability.

## NVIDIA

The single supported candidate is NVIDIA `nvidia/c-foundationstereo-s`, revision
`d776ba855977e99cc9f16246f911d701da0bbc7a`, fixed 320×736 v2 artifact
`onnx/deployable_foundationstereo_small_320x736_v2.0.onnx`, 576104420 bytes,
SHA256 `f3106bd1c8e9c81d5d8ca74f8d21e1e8f111a84abf047f75625fd441253e7ad9`.
Each execution requires license authorization, verified local acquisition and an
actual graph inspection. Model-card/source hints alone do not establish its exact
normalization or names. The inspected pinned graph has float32 `left_image` and
`right_image` inputs `[1,3,320,736]` and float32 `disparity` output `[1,1,320,736]`,
IR8/opset17 and embedded weights.

NVIDIA's [version-specific deployment guidance](https://github.com/nvidia-isaac/foundationpose_perception_pipeline/blob/c1b039726e15dd1038a0e1dcde169dc4dd503088/README.md#L268)
names this exact v2 export. Its [preprocessing](https://github.com/nvidia-isaac/foundationpose_perception_pipeline/blob/c1b039726e15dd1038a0e1dcde169dc4dd503088/src/foundationpose_perception_pipeline/inference/stereo/tao.py#L46)
uses RGB float32 `(RGB/255 - mean)/std`, mean `[0.485,0.456,0.406]`,
std `[0.229,0.224,0.225]`, then batch/channel-first layout. Final disparity is
`x_left - x_right` in input-image pixels; its internal quarter-resolution scaling
has already been applied. The published code has no cryptographic export-source
manifest, so this is supported experimental guidance rather than proven source
lineage. Preserve that limitation in the review. The model card's three-channel
output description conflicts with the observed single-channel graph.

The crop manifest comes from the existing preparation tool. The adapter verifies
the exact 736×320 RGB bytes and shared integer crop/calibration translation without
resize, padding or inferred scale. Reference depth and expected disparity are not
model inputs or outputs. Initial preparation leaves normalization unverified and
execution blocked. After actual artifact inspection and coordinator review,
`prepare --review REVIEW_JSON` accepts these declared review fields:

- `model` exactly matches the pinned metadata in `nvidia_stereo.MODEL`.
- `licenseAuthorization:true`, `normalizationReviewed:true`, and nonempty
  `reviewEvidence` identify the coordinator's authorization/normalization evidence.
- `normalization` is `{mode:raw0_255}`, `{mode:unit0_1}`, or
  `{mode:meanStd,mean:[r,g,b],std:[r,g,b]}` with explicit finite RGB parameters.
- `inputNames:[LEFT_NAME,RIGHT_NAME]` and `outputName` must match actual CPU session
  tensors. Inputs are float32 `[1,3,320,736]`; output is float32 `[1,1,320,736]`.
- `outputQuantity:disparityPixels` and
  `outputConvention:x_left_minus_x_right_in_input_pixels` must be reviewed.
- `inspection:{path,sha256}` pins the real existing inspector report with embedded
  graph checking and no external data. Its path is relative to the review parent.

Those declarations require real coordinator evidence; the adapter cannot prove
human license authorization or normalization semantics from a JSON assertion.

```sh
python scripts/sponsors/nvidia_stereo.py run --plan REVIEWED_PLAN_JSON --model PINNED_LOCAL_ONNX --output NEW_PREDICTION_DIRECTORY
```

Preparation accepts an explicit `--threads 1`, `2` or `4` for CPU intra-op work;
the default is one. Inter-op remains one and the provider remains CPU only.
The complete typed resource limits and CPU configuration enter the plan identity
and cache key. Changing thread count is a new processing experiment, not a retry
of an exhausted attempt. It does not promise faster execution or identical
floating-point results.

The parent bounds one child to 120 seconds, batch one, one job and zero retries.
The child captures one hash-checked graph byte snapshot, recursively
blocks external tensors using the existing inspector traversal, then gives the
same serialized bytes to CPU-only ONNX Runtime and calls `disable_fallback()`.
No memory ceiling or macOS CPU success is claimed. Failed/blocked execution creates
no successful prediction directory. One bounded local experiment of the pinned
artifact completed on macOS arm64 with ONNX Runtime1.24.4 CPU:109.28s for
`session.run`,118.75s complete launch envelope, no model retries. This establishes
that specific execution path; it is not a latency distribution, physical accuracy
result or application/Nebius integration. Raw tiny positive disparities produced
extreme finite depth values and remain in the retained diagnostics.

Successful execution would preserve the raw float32 little-endian disparity and
uint8 numerical validity in the shared prediction-manifest schema. Validity means
finite positive disparity only; it is not confidence, correspondence, surface
support or free space. Local CPU success does not establish Nebius use. Cache reuse
rehashes saved inputs and rasters, and checks model, processing, review and code
identity; it performs no inference. A saved file cannot authenticate its execution
history independently of the coordinator's retained raw provenance. Successful
jobs also preserve child phase markers in `execution-log.jsonl`; a timeout caller
must retain the exception's captured child output to diagnose a failed stage.

Source changes intentionally invalidate plans under a different current reader.
Preserve original plans, predictions and their exact producer source version;
validate saved results with that pinned version and matching package identities.
The original one-thread adapter sources are retained at Git revision `325f31d`.
For existing outputs, call that version's `cached()` directly after verifying the
output exists. `run()` can launch inference when its output is absent. Never
rewrite historical identities or rerun an unchanged model solely to accommodate
a changed reader. The application continues to accept original pinned one-thread
artifacts; new multi-thread artifacts require their explicit matching CPU record.

Cache validation also binds the local model artifact record to the caller's model
path and pinned hash/size, reconstructs both input-tensor hashes from the saved RGB
and reviewed normalization, and validates execution resource counts, UTC timestamp
and nonnegative elapsed time within120 seconds. Declared environment must match
current Python/NumPy/Pillow and installed ONNX/ORT package metadata. This check
does not load the runtime or graph, but missing/mismatched package identity rejects
reuse. Package metadata does not authenticate binaries, and graph-record consistency
does not prove the graph was executed.

## Token Factory

Endpoint `https://api.tokenfactory.nebius.com/v1/chat/completions`, model
`nvidia/Nemotron-3_5-Lightning`. The legacy experimental role
`sceneAdmissionQualityAudit` is rejected as a substantive application integration: its required states and missing-evidence
disposition are already determined by local checks. These helpers are retained
for contract-regression history and have no dispatch or application consumer.
The commands below describe that legacy adapter; they are not the current
integration plan.
Evidence JSON has exact keys `schemaVersion:1`, `observationId`, `evidence`, and
`requiredMissing`. Each evidence item has unique `id`, `state` (`measured`,
`predicted`, or `Unknown`), a short source-grounded `summary`, and a pinned external
`artifact:{path,sha256,bytes,readOnlyExternal:true}`. The coordinator supplies and
verifies evidence summaries; file hashing alone does not verify their interpretation.

Preparation produces `request.json` and `plan.json`, never an audit or prediction.
It requests JSON in the prompt, without assuming per-model `response_format`
support. Bounds: one request, `n:1`, `stream:false`, temperature zero, 512 output
tokens, no retries, 30-second proposed dispatch deadline, 100000 response bytes and
24000 request bytes. This adapter has no transport; the coordinator must enforce
the dispatch deadline/response limit and retain the transport receipt.

```sh
python scripts/sponsors/token_factory.py preflight --plan PLAN_JSON --ledger SHARED_LEDGER --bound VERIFIED_BOUND_JSON --output NEW_PREFLIGHT_DIRECTORY
python scripts/sponsors/token_factory.py accept --plan PLAN_JSON --response RAW_RESPONSE_JSON --receipt TRANSPORT_RECEIPT_JSON --output NEW_AUDIT_DIRECTORY
python scripts/sponsors/token_factory.py cache --plan PLAN_JSON --output EXISTING_AUDIT_DIRECTORY
```

Preflight reads existing ledger fields; it never reserves or mutates them. A bound
receipt supplies exact `model`, `endpoint`, positive integer `maxPromptTokens`,
`contextBoundVerified:true`, `outputBoundIncludesAllBillableGeneration:true`,
`extraChargesBoundVerified:true`, finite `extraChargesUpperBoundUsd`, timezone-aware
`verifiedAtUtc` within the last hour, and exact future `expiresAtUtc`. Those flags
and dates must come from live coordinator verification. Account balance/expiry,
service/pricing, credential presence and no-paid-spill flags must also be verified.
The locked cumulative cap must not exceed min($2,20% verified initial free value),
and usage plus reservations plus the protected demo reserve and new worst-case
request must fit. Account verification must also be within the last hour. A passing
preflight means eligible for coordinator reservation; it does not authorize dispatch
or replace fresh live verification and demo-reserve policy.

Preflight, response acceptance and cache acceptance share one complete prepared-plan
validator. It reconstructs the entire expected contract from captured evidence and
fixed policy, including schema/status/role/endpoint, request, source record, limits,
cost status and full identity. A supplied identity with a recomputed matching cache
hash cannot replace that expected contract. This still validates local consistency,
not provider transport or factuality of evidence summaries.

The transport receipt identifies `source:coordinator_dispatch`, `endpoint`,
`httpStatus:200`, `attempts:1`, `retries:0`, `reservationId`, exact canonical request
SHA256 and raw response SHA256. `billing` supplies `inputTokenUpperBound`,
`inputUsdPerMillion`, `outputUsdPerMillion`, `reservedUsd` and
`extraChargesUpperBoundUsd`. The full input bound plus 512 output tokens and extra
charges must fit the prior reservation. The completion must report the exact model,
request ID, one stopped assistant completion and nonnegative integer token usage.
Truncation, missing usage, unknown citations, duplicate JSON keys, altered input
states and omitted required measurements reject publication. Actual usage is
recorded for coordinator reconciliation, including rejected-response costs outside
this acceptance path. A dispatch failure still requires ledger accounting.

The audit schema has exact keys `observationId`, `disposition` (`unresolved` or
`reviewable`), `checks`, `requiredMissing`. Each supplied evidence ID appears once
with its unchanged state, assessment (`retained`, `flagged`, `unresolved`) and short
reason. Unknown remains unresolved; missing/flagged evidence requires unresolved
disposition. `reviewable` means human measurement review only. Schema/citation checks
cannot prove prose factuality, so the coordinator must review semantic grounding
before substantive application use. This artifact is separate from stereo prediction
manifests and has no geometry, visibility, safety or classification authority.

At the dated $0.06/$0.24 per-million input/output rates, a **verified** 4096-input
token envelope plus 512 output tokens would cost at most $0.00036864 before extra
charges. A **verified** 1048576-input context envelope plus 512 output tokens would
be $0.06303744. These are conditional calculations, not an executable budget or account
authorization. Request byte length is not assumed to bound billed tokens. A separate
coordinator experiment has reached real Nemotron inference through Token Factory
with a verified account-specific envelope. Its completion was truncated and
rejected before scene preparation; no substantive application role is accepted.
That experiment does not validate this legacy audit adapter or close eligibility,
accuracy or coverage requirements.

## Primary API references

- [Token Factory quickstart](https://docs.tokenfactory.nebius.com/quickstart)
- [Token Factory JSON support](https://docs.tokenfactory.nebius.com/ai-models-inference/json)
- [Token Factory API schema](https://api.tokenfactory.nebius.com/docs)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [NVIDIA model card](https://huggingface.co/nvidia/c-foundationstereo-s)

Reverify primary docs, model contracts, event rules and live pricing before execution
or track selection. Analytic tests exercise parser/contracts only; they do not
establish model compatibility, sponsor execution, calibration accuracy or coverage.

The [source-bound evaluation adapter](../evaluation/README.md) checks original
RGB crop/calibration identity, registers genuine disparity back into the original
image grid, and preserves the original inference records separately. It retains
unsupported outside-crop pixels and the full sparse-reference denominator. This
connection performs no inference and does not authenticate a producer receipt
without independently retained execution evidence. Numerical conversion and
development diagnostics do not establish physical accuracy, audit prose grounding,
frontend consumption, Nebius use or hackathon eligibility. Tiny positive disparity
can produce arbitrarily large finite depth; do not treat this numeric mask as
surface confidence or silently clamp the raw output to improve a score.

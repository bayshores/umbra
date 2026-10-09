# Matched development evaluation

This lane replays actual saved CPU StereoSGBM outputs, verifies artifact bindings, and counts unsupported predictions against the full valid source reference. It never declares a release pass. Run with the existing NumPy/Pillow preparation environment; no new dependency, inference or cloud call is needed.

```sh
.local/venv-preparation/bin/python -m unittest discover -s tests/evaluation -v
.local/venv-preparation/bin/python scripts/evaluation/replay_saved.py \
  --source-root . \
  --corpus .local/evaluation/corpus/development-v2.json \
  --corpus-sha256 6df62621610c4fe2ad053594aec9fcafeeabcaa2cdb32432fe3bf748a2f7c563 \
  --baseline-report .local/evaluation/stereo-baseline/five-frame/report.json \
  --baseline-report-sha256 29f68850db8b995f78cf803282c481723c3d7f4bca2db1cd3408fa897f432f35 \
  --output .local/worker-output/evaluation/reproduction
```

Existing outputs are refused. A failed replay can leave partial private artifacts; absence of `report.json` means the run did not finish. Old runs are preserved. Inputs are read-only external references, never symlinked into writable outputs. Raw source inputs, retained prediction/reference arrays and execution-receipt pins are checked before evaluation. Converted float32 depth is byte-equivalent to the saved NPY raster; validity comes from the original recorded matcher support, with fixed-point/conversion agreement checked. Historical float64-score differences are reported with a numerical readback tolerance, separate from physical accuracy gates.

The replay writes one observation manifest, five prediction manifests and result files, explicit float32 depth/uint8 support, per-frame signed float64 errors and pooled raw errors. Five development frames include the previously rejected road-plane scene, whose source-depth diagnostic stays visible with its rejection reason. This does not admit that scene into the application.

For another admitted development prediction, use the same manifests with:

```sh
python scripts/evaluation/matched.py --observations PATH --prediction PATH --output NEW_DIRECTORY
```

## Evaluation detail fields for coordinator compatibility review

The shared version-1 observation/prediction/result envelope is preserved. The following detail fields are specified here for review before adapters adopt them; existing application scene contracts are untouched.

- All artifact records use `path`, `sha256`, `bytes`. Absolute paths and paths resolving outside the manifest parent require `readOnlyExternal:true`. Raster records add `shape:[height,width]` and `encoding`. Raw predictions require `float32le`, support masks `uint8` with exactly 0/1. A PNG reference uses `uint16png`, an explicit positive `divisor` and zero/nonfinite/nonpositive values remain invalid. No max-range filtering removes valid references.
- `sources` is a list of artifact records with unique `role` strings. Every prediction source role/hash must occur in its observation, and bytes are verified at both pointers. Source bytes for every observation are verified before auditing the split, including observations outside the scored frame. Roles `left`, `right`, `image` participate in cross-split duplicate-image checks; calibration reuse is not duplicate-image leakage.
- `processing` declares `originalDimensions`, `inputDimensions`, `outputDimensions` in height/width order, `crop`, `pad`, `resize`, `normalization`, `coordinateConvention`, `depthConvention`. This first adapter admits exact full-frame identity dimensions, null crop/pad/resize, `OpenCV-left-rectified-x-right-y-down-z-forward`, and `axialZ` only. Shape coincidence alone cannot admit a crop/resize or radial-depth conversion.
- `reference` requires `raster`, the same explicit `processing`, `independent`, `method`, `timing`, `uncertainty` and `limits`. The raster quantity is `axialDepthMeters`. `calibration.artifact` binds its actual source bytes. Disparity predictions are refused until an upstream source-pinned conversion adapter explicitly establishes the corresponding metric raster. The saved-output adapter retains the original conversion/calibration receipt.
- The prediction's `model` requires `provider`, `id`, and `version` or `revision`; `execution` requires `status:succeeded`, `codeHashes`, `environment`. The saved adapter retains the original actual inference receipt and marks adaptation as reuse with zero new inference/cloud calls. StereoSGBM configuration is pinned as `parametersHash`; it is not mislabeled as learned model weights. Assertions in manifests are provenance declarations requiring independent review, not authentication of model execution or calibration.
- Observations use `siteGroup` for conservative leakage grouping. Optional `physicalSite` is null or an object with `id` and `evidence`; it never defaults to a frame ID. `developmentExposed` and `modelExposure` (`verifiedAbsent` only when evidenced) are holdout audit details. Conflicting sequence/site/image identities across development/unassigned and heldout candidates reject evaluation. Unknown sites or model exposure cannot establish holdout independence. Scoring any non-development observation is refused until the coordinator supplies a justified reviewed lock/reference design in a future change.
- Optional `--visibility-cases PATH` accepts a version-1 manifest with `observationId`, `predictionManifestHash`, a hashed `predictionEvidence` artifact, and `cases`. Each case has `id`, `prediction` (`Blocked`, `Clear`, `Unknown`), `reference` (`Blocked`, `Clear`, `unresolved`), `independent`, `referenceEvidence` (hashed artifact list), `independenceGroup`, and optional `condition`. Only independent evidence-backed labels are resolvable. Unknown on an unresolved label is never scored correct. Counts include full confusion, all/resolveable Unknown, false Clear on verified Blocked, independent groups, risk, useful correct decision coverage and condition strata. No blocked reference means false-Clear count is null; no independent sampling design means its confidence bound is null.

Each result preserves `thresholdLock:null`, source/code/manifest hashes, admission/rejection, explicit limitations, source-support and full-reference denominators, depth error distributions and reference-distance strata. Fixed 1/2m and positive 10/100m development diagnostic bins reproduce earlier studies; they are not locked physical tolerances. Raw errors are ordered by row-major valid reference AND supported prediction pixels.

## Evidence limits and release gaps

The current source is sparse, temporally fused and stereo-model-guided filtered LiDAR with provisional calibration/axial-Z interpretation. Metrics against it do not establish independent depth accuracy, ground/footprint continuity, free space, whole-ray visibility or safety. The five frames share a sequence; distinct physical-site count is unknown. California sites, independent metric references and final holdout observations remain zero. No real visibility labels exist in this replay, so false Clear is unmeasured and useful physical coverage is null. Analytic test fixtures cannot close any of those gaps.

There is no statistically justified false-Clear confidence bound, useful-coverage lock or automated release gating in this milestone. The coordinator must establish task decision resolution, independent reference precision, real blocked/clear labels and site/condition sampling adequacy before adding final heldout execution. Provenance reads are sequential, not an atomic multi-file snapshot or proof of loaded-code identity. Fresh independent review is still required.

Performance is deliberately unmeasured during concurrent work. After all workers stop, the coordinator can profile a fresh replay in isolation with `/usr/bin/time -l` and the command above, retaining hardware/environment and cold/warm conditions. That measures this saved-output evaluator only; it cannot substitute for actual inference/load/browser performance.

## Explicit disparity registration

`disparity_registration.register` is a pure numerical operation, with explicit float32 disparity, binary uint8 support, captured calibration bytes, original `[height,width]`, equal integer left/right crop `[x,y,width,height]` and left-minus-right disparity convention. The existing strict rectified parser derives the scaled projection numerator and principal-point offset from source matrices. Conversion uses float64 division and float32 storage. Unrepresentable, nonfinite and nonpositive depths are unsupported. Equal crop translation preserves disparity/offset; embedding keeps NaN/mask0 outside the crop and performs no interpolation or filling.

The returned derivation separates converter-input crop dimensions from registered full-grid dimensions. Full-reference evaluation must count supported reference pixels outside that crop as invalid prediction. This module authenticates neither the producer nor inference/source identities. A caller must verify those contracts separately and retain original inference dimensions/processing distinctly; declaring full-grid processing cannot erase an inference crop or turn a post-inference crop into model output. Numerical support does not imply confidence, calibration accuracy, free space or physical visibility. This operation is not an NVIDIA or Nebius execution adapter.

Run the labeled analytic numerical checks with `python -B -m unittest discover -s tests/evaluation -p test_disparity_registration.py -v`. Fixtures are excluded from runtime evidence.

## Bind a saved NVIDIA prediction to a development observation

`adapt_nvidia.py` consumes an existing prediction; it never acquires artifacts or invokes inference/cloud services. Its required arguments are `--observations`, `--observations-sha256`, `--plan`, `--plan-sha256`, `--prediction-dir`, `--prediction-sha256`, `--model` and `--output`. Hashes pin the exact observation manifest, prepared plan and producer `prediction.json` selected for review. Missing successful predictions or unreviewed model prerequisites stop the branch, with no successful result published. An existing destination is retained.

The adapter reuses the NVIDIA cache validator, verifies the crop against original left/right images and source calibration, then converts/embeds its disparity explicitly. Original producer processing/sources/execution remain separately recorded in the derivation with their original manifest base. Registered output is evaluated on the full original supported reference, including reference pixels outside the crop as invalid prediction. Only development observations are permitted; source/reference validity and split checks are inherited from the evaluator. Reference depth never supplies model predictions.

Outputs are registered `prediction.json`, float32 depth, uint8 support, `result.json` and raw signed errors. Publication is exclusive after staged evaluation. Cache consistency and successful status fields are declarations requiring independent genuine-execution review; this command does not authenticate historical computation or establish Nebius/track eligibility. Tests in `test_prediction_binding.py` use isolated analytic fixtures, including an explicitly mocked cache validator for publication checks; those files are never runtime/model evidence.

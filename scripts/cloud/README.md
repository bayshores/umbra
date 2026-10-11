# Nebius Job control evidence

`job_evidence.py` is a standard library collector for one ordinary authenticated
Job Get from `https://api.nebius.cloud/ai/v1/jobs/{id}`. It records successful Job
control completion, with `claim: job_control_only`, `modelExecutionVerified: false`
and `outputsBound: false`. It does not validate model execution, output artifacts,
execution attempts, sponsor eligibility or application integration. The
consistency parser alone cannot authenticate a response. No receipt contains an
`authenticated` flag or fabricated attempt identity.

The collector creates no resources, launches no work, follows no redirects,
retries no request and requests no SECRET view. It uses the fixed official host,
default TLS certificate/hostname verification, no environment proxy settings,
a configured 30-second socket timeout and POSIX main-thread wall alarm, and a
2 MiB response-body limit. HTTP errors are rejected before their body is read.
The deadline refuses an already active process alarm without changing it and
restores the previous signal handler on completion/failure. Platforms without
POSIX file/signal features fail closed.

An offline check verifies that a short alarm interrupts a real sleep and
restores signal state. Native resolver behavior, TLS and provider slow-path
interruption have not been exercised; a hard end-to-end provider time bound is
unverified. These configured limits are not a provider spending cutoff.

The source schema is [Nebius Get](https://docs.nebius.com/rest-api/ai/v1/jobs/get),
with [Job definitions pinned at vendor commit
97201c925d3b03924ee4d1020a4ce31ab35f83e8](https://github.com/nebius/api/blob/97201c925d3b03924ee4d1020a4ce31ab35f83e8/nebius/ai/v1/job.proto).
`containerCommand` and `args` are strings. Job `COMPLETED` is enum integer 6;
both that exact integer and the name are accepted under the
[ProtoJSON enum rules](https://protobuf.dev/programming-guides/json/).
Instance enum integer 7 is a different field and is not accepted as Job success.

After a separately authorized genuine submission, the coordinator must freeze
one non-secret expected JSON file with exactly these fields:

```json
{
  "schemaVersion": 1,
  "jobId": "actual-submitted-job-id",
  "projectId": "actual-project-id",
  "image": "registry.example/approved-image@sha256:<64-lowercase-hex-digest>",
  "containerCommand": "actual approved command",
  "args": "actual approved arguments"
}
```

This is a schema illustration, not a valid expected file or submission record.
The image must be digest-pinned. Identities reject traversal and URL/control
characters. Command/arguments must be printable non-secret ASCII strings.
Metadata identities and all three spec fields must match exactly. UTC `Z`
created/start/finish timestamps must be ordered and no later than acquisition;
comparison retains all nine protobuf fractional digits and saves original
timestamp strings. JSON rejects duplicate keys, nonfinite values and oversized
input. A matching requested image digest does not attest the resolved image or
its inputs/outputs. A resource version is not an execution attempt.
Resource versions accept only nonnegative int64 integers or decimal strings
and preserve the returned representation; an omitted version is saved as
`null` (unknown). ProtoJSON may omit empty command/argument defaults: omission
matches only a frozen empty string, while image omission is rejected.

When live collection is explicitly authorized, supply an AI Cloud bearer token
in a separate regular, current-user-owned, non-symlink mode0600 file. Token
Factory credentials are not automatically AI Cloud credentials. The CLI accepts
only a token **file path**, with no environment variable, inline token or host
override. Do not put credentials in expected command/arguments. Do not create
or inspect real credential files merely to try this script.

```sh
python3 scripts/cloud/job_evidence.py \
  --expected .local/approved-job-expected.json \
  --token-file /explicit/private/ai-cloud-token-file \
  --output .local/approved-job-control.json
```

Output parent directories must already exist inside this project's `.local/`
and may not be symlinks. The collector reserves an exclusive mode0600 regular
file before reading credentials or contacting the service. Existing files and
symlinks fail without overwrite or credential/network access. Failed collection
removes its own reserved file. CLI errors disclose generic redacted text only.

The receipt whitelists metadata id/parent/resourceVersion/createdAt, spec
image/command/arguments and status state/start/finish. It omits instances,
addresses, endpoints, environment variables, injected content, registry/S3
credentials, labels, names, logs, service error text and raw response bytes.
Provenance includes acquisition UTC, the collector source SHA256, exact
expected-file SHA256, response-byte SHA256 and request SHA256. The request hash
covers canonical method, fixed URL and non-secret Accept/Accept-Encoding headers;
it deliberately excludes the Authorization credential. These hashes are local
integrity records, not provider signatures or output attestation. A saved file
can still be copied/edited externally; it is not automatic runtime authority.

Offline validation uses temporary analytic HTTP/JSON fixtures and no network:

```sh
python3 -m unittest discover -s tests/cloud -p 'test_*.py'
```

Tests and this collector alone do not admit remote predictions to cached readers or
the app, change existing local sponsor evidence, or close the integration gate.
No genuine successful receipt, expected job identity or provider credential is
provided by this change. Submission, billing/credit checks, exact inputs/model
bindings, job-associated application execution/logs, retrieved output hashes and
consumer validation remain separately required before remote output acceptance.

## One-job application output binding

`run_stereo_job.py` is the explicit CPU workload wrapper for a separately approved
job. It creates no cloud resources and makes no service API calls. It refuses to
execute without `--execute`, reserves a new private output directory, verifies
the supplied exact plan SHA256, and invokes the existing NVIDIA adapter once.
The adapter's 120-second child bound and zero-retry policy remain unchanged.
Existing output or an adapter cache-race return is rejected. It then validates
the result in the same process and preserves the original CPU producer bytes.

The output contains exact `plan.json`, the four original files in `prediction/`,
and `job-result.json`. The binding records the unique 32-hex submission identifier,
plan/model/cache/code/tensor identities, wrapper source hash and output hashes.
It claims only `application_output_binding_only`. The CPU producer remains a
record of the process inside the workload; it is not relabelled as authenticated
Nebius execution. Inference or binding failure retains available raw child error
bytes and never leaves this wrapper's success marker. A later retrieval failure
may retain a valid application-only binding, but establishes neither delivery nor
a successful provider Job. The script accepts no credentials or billing flags.
Use `python3 scripts/cloud/run_stereo_job.py --help` to inspect required arguments;
execution is not a setup check and needs the project's separate authorization.

For an ephemeral Serverless Job, `--retrieval-window-seconds 600` keeps a
successful result available for at most ten minutes. After a controlled download
and offline validation, the coordinator creates the empty regular file
`/output/.<submission-id>-retrieved` outside the result bundle. The wrapper then
exits normally so the Job can reach `COMPLETED`; cancelling a successful job
would instead fail the existing Job evidence check. A pre-existing acknowledgement
is refused before inference. The acknowledgement must be an empty regular
non-symlink file owned by the wrapper's own user (root in the current image), so
create it from inside the container as its default user, for example
`sudo docker exec -i <container> python -B -` with an exclusive `os.open`. The
wrapper polls about once per second and rejects one it first observes after the
window ends. No acknowledgement within the window is a failure,
not a successful delivery. Zero (the default) preserves ordinary local execution.
The runtime contract's optional `retrievalWindowSeconds` pins this exact flag.
The provider timeout and coordinator cancellation deadline remain separate
controls; this acknowledgement window is not a provider billing cutoff.

`result_binding.validate_binding(bundle_root, expected_submission_id,
expected_plan_sha256, expected_binding_sha256=...)` returns the original producer
and binding after bounded offline consistency checks. It reads exactly the four
flat output files, rejects symlinks/path escapes/strict-JSON violations and checks
full raw raster sizes and numerical validity. It preserves invalid pixels and
raw numerical tails. Comparison of two recorded tensor hashes is not independent
tensor reproduction, and environment strings do not authenticate installed binaries.

Remote admission still requires trusted authenticated Job control and exact
submitted configuration, the job-associated controlled artifact retrieval,
independently captured input/review/tensor identities, the approved image/runtime
record, and the explicit attachment/application path. Requested image digests and
local file hashes are not provider signatures. Neither this offline validator nor
analytic fixture tests establish sponsor use, model accuracy or a usable crossing.
The existing local cache reader and sponsor source hashes remain unchanged.
No real service output is included with these tools.

## Direct-native consumption and attachment

`consume_result.check_result(bundle_root, **pins)` reproduces both input tensors
from independently pinned native images and validates calibration, review,
processing, model and CPU configuration through the unchanged adapter. A private
temporary validation plan changes only reference paths; the retrieved plan and
CPU producer remain byte-for-byte originals. The expected submitted command and
arguments must match the exact wrapper invocation, submission identifier and
plan hash. A separately pinned runtime contract fixes the requested image,
wrapper hash, five environment versions and absolute workload paths. Environment
strings describe the workload record; they do not attest installed binaries.

`consume_result.collect_result(bundle_root, token_path=..., receipt_path=...,
**pins)` completes this preflight before credentials or network access, freezes
the captured expected-job bytes, then invokes the existing direct collector
once. It checks the returned job, configuration and exact timestamp interval,
including nanoseconds, and rechecks captured artifacts. It returns the original
producer, application output binding and separate service context with claim
`job_control_and_application_output_association`. A failure after collection
retains the genuine control receipt but returns no accepted association.
Check-only output has claim `offline_consistency_only` and cannot admit a remote
result. The CLI defaults to check-only; `--collect` is an explicit service read,
never job creation, inference or a retry.

The independently pinned inputs are `native_manifest_path`, `review_path`,
`expected_job_path` and `runtime_contract_path`, each with its expected SHA256,
plus `expected_submission_id`, `expected_plan_sha256` and optional output-binding
SHA256. `scratch_root` optionally selects a private project-local temporary root.
Use `python3 scripts/cloud/consume_result.py --help` for CLI spellings. Preserve
controlled retrieval of the actual job's output bundle as a separate prerequisite;
an arbitrary local bundle with matching files does not prove remote origin.

`scripts/california/attach_av2_prediction.py --direct-native` retains its existing
local-cache behavior by default. Explicit remote mode additionally takes paired
`--nebius-options` and `--nebius-options-sha256` arguments. This pinned private JSON
request contains the consumer pins above (native manifest and plan pins come from
the existing attachment arguments), plus `token_path` and fresh `receipt_path`.
It contains credential file paths only. Remote mode rejects source/bundle
symlinks, completes native measured and render checks before collecting, then
preserves the original producer and plan and adds separate hashed context,
control and binding artifacts. Its attachment report records one job-control
request and zero new model runs. The original CPU record keeps its own zero
service requests, describing the process inside the workload.

The application checks these compiled artifacts against the reconstruction
identity at scene load. It performs no provider authentication or paid inference
in the browser. Predicted points remain renderer-only, measured buffers retain
query ownership, and missing ground evidence stays unvalidated. Ordinary dragging
and comparisons remain local. Analytic transport and attachment tests establish
these code paths, not genuine sponsor execution, provider output signatures,
physical accuracy, usable crossing coverage or hackathon eligibility.

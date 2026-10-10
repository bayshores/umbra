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

Tests and this collector do not admit remote predictions to cached readers or
the app, change existing local sponsor evidence, or close the integration gate.
No genuine successful receipt, expected job identity or provider credential is
provided by this change. Submission, billing/credit checks, exact inputs/model
bindings, job-associated application execution/logs, retrieved output hashes and
consumer validation remain separately required before remote output acceptance.

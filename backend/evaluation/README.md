# Research evaluation

These tools measure the deployed pipeline. They do not reproduce the paper's old
30-epoch experiments and do not establish agronomic correctness.

## Offline advisory smoke

From the repository root, use the environment containing the backend dependencies:

```sh
backend/.venv/bin/python backend/evaluation/llm.py --output backend/evaluation/runs/rules-01
```

This runs 17 cases (four classes × two languages × two confidence levels, plus an
ambiguous two-disease case). It makes **no LLM calls**. Metrics without observations
are null, not evidence of provider success. Output directories must be new.

## Live LLM comparison

Set keys/model IDs in backend/.env, configure at least two providers, then:

```sh
AI_MAX_REQUESTS_PER_HOUR=300 backend/.venv/bin/python backend/evaluation/llm.py --live --repeats 1 --output backend/evaluation/runs/live-01
```

Live mode compares rules, each configured provider alone, and consensus on the same
cases. The cache is cleared before every observation. Retries consume the configured
attempt budget; provider-side quotas still apply. This command can consume credits
on a paid account/model. It does not reset the attempt budget between cases.

Outputs:
- `responses.jsonl`: case settings, result text, source, votes, latency, request ID.
- `audit.jsonl`: individual provider attempts, parse failures, retries, filtered items.
- `summary.json`: overall and grouped metrics, model IDs, code/protocol hashes, budget.
- `expert-review.csv`: blank 0–4 correctness/safety/local relevance fields for a named
  agronomist to fill. Automated protocol compliance is not clinical/agronomic correctness.

Report failures, quota exhaustion and fallback rates alongside latency; do not rank a
provider from latency alone. The case set is a small protocol test suite, not a field trial.
Use multiple repetitions for variance and a balanced, externally reviewed case set before
publishing model comparisons. Raw output logging remains off unless RESEARCH_LOG_RAW=1.

## Classification smoke and common-set benchmark

```sh
backend/.venv/bin/python backend/evaluation/classification.py --manifest backend/evaluation/sample-manifest.json --smoke --output backend/evaluation/runs/images-smoke-01
```

This exercises all four individual models, the four-model mean, and the EfficientNet-B0
→ ResNet50 cascade on the included samples. The output is explicitly marked
`smoke_not_accuracy_evidence`; do not publish the four-sample scores as test accuracy.

For a real evaluation, create a JSON array with `{path, label, sha256}` for each image.
Paths resolve relative to the manifest. Labels use the four canonical class names.
Supply a separate JSON object mapping **all four model IDs** to complete lists of
SHA-256 hashes of their training images. Then run without --smoke:

```sh
backend/.venv/bin/python backend/evaluation/classification.py --manifest /path/common-test.json --training-hashes /path/training-hashes.json --threshold 0.8 --output backend/evaluation/runs/common-test-01
```

The tool rejects duplicate images, checksum mismatch, unsupported labels, missing
class coverage, and exact image overlap with any supplied training inventory.
Hash matching cannot detect crops/augmentations or photos of the same plant; the
researcher must independently prevent those leaks and verify the inventories are complete.
The original preprocessing has not been recovered; ImageNet normalization remains an
assumption and must be validated before claiming equivalence to training.

Outputs contain per-image predictions and a summary with confusion matrices,
per-class precision/recall/F1, macro F1, accuracy, median latency, dataset hash, checkpoint
manifest, threshold and code hash. Every mode sees the same images. Latency includes
loading with the application's one-model cache, so it is not a warm-inference benchmark.

## Research audit logs

The backend writes local rotating JSONL (5 MB + 3 backups) by default. Set
RESEARCH_LOG_PATH to persistent storage on Render to retain logs across deployments;
relative paths resolve under backend/. Use one process per path, or distinct paths
for multiple workers. No log download endpoint is exposed.

Classification records include model IDs, scores, timing and a prediction ID (no image
or filename). Advisory/provider events share a request ID and link to prediction_id
when supplied. Cached advisory events are marked; they do not fabricate provider calls.
The prompt and protocol have hashes. Keys, request headers and raw exception messages
are never logged. Optional raw model text is truncated and known keys redacted; keep
this off for ordinary use. Log storage failure does not fail a user's diagnosis.

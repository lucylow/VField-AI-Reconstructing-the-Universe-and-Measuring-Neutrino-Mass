# Benchmark result schema

A benchmark report should be a JSON object containing these fields. Additional fields are allowed but
must not silently redefine existing keys.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | version of report schema |
| `physics_level` | enum | `toy`, `approximate`, or `validated`; current runner is `toy` |
| `scientific_warning` | string | mandatory warning for prototype results |
| `method.name` | string | stable method identifier |
| `method.config` | object | hyperparameters and preprocessing |
| `dataset.sha256` | string | hash of exact dataset artifact |
| `split.train_ids` | array | training realization IDs |
| `split.validation_ids` | array | validation IDs |
| `split.test_ids` | array | untouched test IDs |
| `parameter_names` | array | parameter ordering |
| `prior` | object | prior support and density convention |
| `bias` | map | mean posterior point-estimate error per parameter |
| `mae` | map | median absolute error per parameter |
| `coverage` | map | central interval inclusion frequency per parameter and level |
| `interval_width` | map | average/median interval width |
| `ess` | number | effective particle count summary when applicable |
| `runtime_seconds` | number | measured, not guessed, runtime |
| `peak_memory_mb` | number/null | observed peak memory if measured |
| `limitations` | array | interpretation boundaries |

Report coverage with binomial uncertainty. A 68% nominal interval measured on ten test examples is
not evidence of calibration. When evaluating multiple parameters and many interval levels, preserve
all metrics and avoid selecting only the favorable plots for publication.

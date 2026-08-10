# Local benchmark harness

`benchmarks.run` is a small development harness for checking the Retrieval Lab
runner locally. It generates a deterministic synthetic document corpus and
graded document-level qrels in memory, then evaluates `keyword` and `bm25`
under the same conditions. It does not call a service, download a model, or
require network access.

## Method

The harness uses a fixed dataset identity (`retrieval-lab.synthetic`, version
`1`) and a seed. The small tier contains a bounded corpus and query set; the
medium tier is larger but uses the same generator and remains practical on a
developer machine. Each query has one graded relevant document. This ground
truth is synthetic and is not a quality claim about a real collection.

Both phases run in the same Python process and use the same temporary cache:

- `cold` starts with an empty cache and records cache misses/rebuild time.
- `warm` runs immediately afterwards and records cache hits.

The harness checks that both phases produce the same deterministic `run_id` and
aggregate metrics. Runtime timings, timestamps, and cache status are report
observations only; they are not inserted into the Retrieval Lab run identity or
comparison contract.

## Run it

From the repository root:

```console
uv run python -m benchmarks.run --size small --output /tmp/retrieval-lab-small.json
uv run python -m benchmarks.run --size medium --output /tmp/retrieval-lab-medium.json
```

Optional controls are `--seed`, `--top-k 1,3,5`, and `--repetitions 1`.
The current v0.1 runner is intentionally single-run, so repetitions other than
one are rejected rather than silently approximated.

The report is written with a same-directory temporary file, flush, `fsync`, and
atomic replacement. Existing output symlinks and symlinked path components are
rejected. The JSON writer rejects non-finite numbers and the report loader
rejects duplicate keys and `NaN`/`Infinity` constants.

## Report shape

The versioned report contains:

- benchmark dataset ID/version, tier, seed, corpus/query counts, retrievers,
  top-k, relevance level, and repetitions;
- OS system/release/machine, CPU architecture/description, Python version, and
  the installed Retrieval Lab version;
- UTC start/finish timestamps and wall-clock duration;
- cold and warm run IDs, aggregate metric maps, existing per-retriever latency
  statistics, index sizes when available, and a sanitized cache-event summary;
- explicit equality checks for cold/warm run IDs and metrics.

Query text, document text, retrieved IDs, cache paths, absolute paths, user or
host names, environment values, and secrets are intentionally omitted. The
runtime `build_ms` values are copied from the existing result manifest, while
the aggregate metric and latency maps come from the existing
`EvaluationResult`.

## Reading results

Use the cache status and build timing to distinguish setup work from search
latency. A warm run is not a claim about a production cache: it only describes
the second phase in this one process. Compare runs only when the deterministic
dataset and evaluation settings are compatible.

Search p95 is a nearest-rank estimate. Small query samples can make p95
unstable; the existing latency warning is preserved in each retriever's
latency record. Do not use this synthetic harness to publish throughput,
tail-latency, hardware, or production-quality claims. CPU, OS, Python version,
cache state, and workload size all affect observations.

## Constraints

This is an offline smoke benchmark, not a load generator. It exercises only
the public Python API and the dependency-free keyword/BM25 implementations.
Dense model downloads, external retrievers, concurrency, repeated-run
statistics, and production corpora require a separately designed experiment.

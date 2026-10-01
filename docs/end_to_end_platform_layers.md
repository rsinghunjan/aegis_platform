# End-to-End AI Platform Layers

This document describes the production-grade platform layers added to turn
Aegis from a supervised agentic control plane into a complete end-to-end AI
platform. Each layer is additive: it does not modify the existing
`agentic/runtime.py` control plane, and every module imports optional
third-party SDKs lazily so the packages can be imported in minimal
environments (tests, CI) without those SDKs installed.

## 1. Inference layer — `services/inference/`

- `registry.py` — thread-safe `ModelRegistry` with per-name version history
  and content-addressed fingerprints for audit trails.
- `providers.py` — `InferenceProvider` abstraction with implementations for
  a dependency-free local `EchoProvider` (safe default/test double),
  `OpenAICompatibleProvider`, `AnthropicProvider`, `OllamaProvider`, and
  `VLLMProvider`.
- `router.py` — `ModelRouter` tries providers in order and falls back to
  the next one on failure/unavailability, recording each attempt.
- `tokens.py` — token counting (via `tiktoken` if installed, else a
  whitespace/character heuristic) and USD cost estimation.
- `batch.py` — `BatchInferenceQueue`, a bounded worker-pool queue for
  decoupling request submission from (potentially slow) model calls.

## 2. Vector & embeddings — `services/embeddings/`

- `providers.py` — `EmbeddingProvider` abstraction: dependency-free
  `LocalHashEmbeddingProvider` (deterministic hash-bucket embeddings used
  as the default/test fallback), plus `OpenAIEmbeddingProvider` and
  `CohereEmbeddingProvider`.
- `vector_store.py` — `VectorStore` abstraction: `InMemoryVectorStore`
  (brute-force cosine similarity, with TTL expiry) and `PGVectorStore`
  (pgvector-backed). Additional backends (Milvus, Pinecone, Weaviate,
  Qdrant) can implement the same interface.
- `chunking.py` — sentence-boundary-aware semantic chunking with
  configurable overlap.
- `rag.py` — `RAGPipeline` ties chunking + embedding + vector store
  together for document ingestion, retrieval, and prompt augmentation.

## 3. Data infrastructure — `services/data/`

- `lake.py` — `DataLake` abstraction: `LocalDataLake` (filesystem,
  path-escape protected) and `S3DataLake`.
- `warehouse.py` — `WarehouseConnector` abstraction: `DuckDBWarehouse`
  (embedded OLAP, no external service required), `BigQueryWarehouse`,
  `SnowflakeWarehouse`.
- `feature_store.py` — `InMemoryFeatureStore` with online (latest value)
  and offline (point-in-time history) reads, mirroring the Feast/Tecton
  access pattern.
- `quality.py` — lightweight Great-Expectations-style checks
  (`expect_not_null`, `expect_between`, `expect_in_set`) and
  `run_expectations` producing a `DataQualityReport`.
- `catalog.py` — `MetadataCatalog`, an in-memory DataHub-style dataset
  registry with tag search and upstream lineage lookup.
- `etl.py` — `ETLPipeline` runs a DAG of `ETLStep`s in dependency order,
  parallelizing independent steps via a thread pool.

## 4. Cloud integrations — `services/cloud/`

- `provider.py` — `CloudProvider` abstraction for AWS, GCP, Azure, OCI,
  and Alibaba Cloud (instance listing, cost & usage where supported).
- `cost.py` — `CostTracker` with per-scope budgets; raises
  `BudgetExceededError` when cumulative spend crosses a configured limit.
- `secrets.py` — `SecretManager` abstraction: `EnvSecretManager` (safe
  default), `VaultSecretManager`, `AWSSecretsManager`, `GCPSecretManager`.
- `deploy.py` — `ModelDeployment` lifecycle manager (pending → deploying →
  running/failed → terminated) across `DeploymentTarget`s (EC2, Cloud Run,
  ACI, Kubernetes, local).
- `autoscale.py` — threshold-based `AutoScalingPolicy` (mirrors a
  Kubernetes HPA's scale-up/scale-down/hold decision logic).

## 5. Advanced reasoning — `agentic/reasoning/`

- `decomposition.py` — heuristic sequential goal decomposition (numbered
  lists, "then"/"and then", semicolons), with an optional
  model-backed `decomposer_fn` override.
- `prompting.py` — chain-of-thought and tree-of-thought prompt builders.
- `uncertainty.py` — `score_confidence` combines self-consistency across
  repeated samples, hedging-language detection, and response-length
  heuristics into a bounded confidence score.
- `refinement.py` — `PlanRefiner` runs an iterative critique/revise loop
  until the plan converges or a max-iteration bound is hit.
- `bandit.py` — `EpsilonGreedyBandit` and `UCB1Bandit` for tool-selection
  optimization over repeated runs.
- `verification.py` — `verify_plan` runs structural checks (missing tool,
  disallowed tool, duplicate steps) and `backtrack_to_last_valid_step`
  truncates a plan at the first fatal error.

## 6. UI dashboard — `frontend/`

A React + TypeScript dashboard (Vite-based) with:

- `RunMonitor` — polls agent runs and displays status.
- `ApprovalQueue` — lists pending approvals with approve/deny actions.
- `EvidenceExplorer` — renders a run's evidence hash chain, flagging
  chain breaks.
- `AnalyticsPanel` — aggregate run counts by status.

See `frontend/README.md` for development and build instructions.

## 7. Observability — `services/observability/`

- `tracing.py` — `Tracer`/`Span` with a dependency-free in-process
  implementation recording completed spans with parent/child linkage and
  duration; ready to be swapped for a real OpenTelemetry exporter.
- `logging.py` — `JSONFormatter` for structured single-line JSON logs plus
  a `contextvars`-based correlation id propagated across nested calls.
- `metrics.py` — `MetricsRegistry` with counter/gauge/histogram
  primitives and Prometheus-text-format rendering, usable standalone or
  alongside `prometheus_client`.
- `health.py` — `HealthRegistry` for liveness/readiness aggregation and a
  `CircuitBreaker` (closed → open → half-open) for resilient external
  calls.

## 8. Production hardening — `ops/production/`

- `helm/aegis/` — Helm chart with `Deployment` (readiness/liveness
  probes), `Service`, `Ingress` (cert-manager annotations, TLS),
  `HorizontalPodAutoscaler`, and `PodDisruptionBudget` templates.
  Validated with `helm lint` and `helm template`.
- `.github/workflows/platform-layers-ci.yml` — CI pipeline that runs the
  new layers' unit tests, lints/templates the Helm chart, and
  type-checks/builds the frontend.

## Testing

Each layer has a corresponding unit test module under `tests/`:

```
tests/test_inference_layer.py
tests/test_embeddings_layer.py
tests/test_data_layer.py
tests/test_cloud_layer.py
tests/test_reasoning_layer.py
tests/test_observability_layer.py
```

Run them with:

```bash
pip install -r requirements-control-plane.txt
pip install pytest python-multipart httpx duckdb
python -m pytest tests/test_inference_layer.py tests/test_embeddings_layer.py \
  tests/test_data_layer.py tests/test_cloud_layer.py \
  tests/test_reasoning_layer.py tests/test_observability_layer.py -q
```

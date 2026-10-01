 
```markdown
Model runtime wiring & serving guide (Aegis)

Files added:
- api/model_loader.py       (PyTorch & ONNX wrappers + warmup)
- api/batcher.py            (async batching engine)
- api/model_runner.py       (ModelRegistry with batching+concurrency)

Quick integration (minimal)
1. Install runtime deps (choose variant for GPU):
   pip install -r api/requirements-models.txt
   # For GPU PyTorch, install torch with CUDA per PyTorch docs.

2. Register a model (example):
   from api.model_runner import registry, ModelConfig
   registry.register(
       model_name="multimodal_demo",
       version="v1",
       config=ModelConfig(
           model_path="/path/to/artifact.pt",
           runtime="torch",
           device="cuda",                  # or "cpu"
           max_concurrency=4,
           max_batch_size=8,
           batch_latency_ms=50,
           warmup_sample={"tensor": [0]*128},  # adapt to your input shape
           warmup_iters=2
       )
   )
   registry.load("multimodal_demo", "v1")

3. Use in your API handler (example FastAPI):
   from api.model_runner import registry
   def predict_handler(...):
       result = registry.predict_sync("multimodal_demo", "v1", input_payload)
       return result

Notes & best practices
- Preprocess inputs (text tokenization, image resize -> tensors) before passing to registry; model wrappers expect tensors/arrays or dicts keyed by input names.
- For large files, stream from object store to local temp file and pass path/stream to preprocessing step; avoid embedding large base64 in HTTP payloads.
- Tune max_batch_size and batch_latency_ms for throughput vs latency trade-off.
- Monitor GPU memory usage (torch.cuda.memory_reserved) and configure cpu_offload / eviction policies for low-memory situations.
- Add Prometheus metrics around predict latency, batch size, queue length and concurrency permits.
- For production, implement:
  - a model eviction LRU with memory watermark triggers
  - scheduling of GPU workers for heavy models (separate kube node pools)
  - request prioritization (small/interactive vs batch jobs)
```

 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
"""
    try:
        model_path = meta.path if meta else None
        model_wrapper = ModelWrapper(model_path=model_path, model_version=MODEL_VERSION)
        model_wrapper.load()
    except Exception:
        logger.exception("Model loading failed; model_wrapper is not available")
        model_wrapper = None

    # Setup batcher if enabled and model supports batch predict
    if BATCH_ENABLED and model_wrapper is not None:
        batcher = BatchWorker(predict_fn=model_wrapper.predict, max_batch_size=BATCH_MAX_SIZE, max_latency_s=BATCH_MAX_LATENCY_MS / 1000.0)
        batcher.start()
        logger.info("Batching enabled (size=%d, latency_ms=%d)", BATCH_MAX_SIZE, BATCH_MAX_LATENCY_MS)
    else:
        logger.info("Batching disabled or model not loaded")

@app.on_event("shutdown")
async def shutdown_event():
    if batcher:
        try:
            await batcher.stop()
        except Exception:
            logger.exception("Error stopping batcher")


@app.get("/health")
def health():
    """
    Return structured health including:
    - model loaded and version
    - registry health summary
    - batching configuration & queue size
    """
    registry_health = registry.health()
    model_info = {"loaded": bool(model_wrapper and getattr(model_wrapper, "model", None) is not None), "model_version": getattr(model_wrapper, "model_version", None) if model_wrapper else None}
    batching = {"enabled": bool(batcher), "max_batch_size": BATCH_MAX_SIZE, "queue_size": batcher._queue.qsize() if batcher else 0}
    return JSONResponse(status_code=200, content={"ok": True, "registry": registry_health, "model": model_info, "batching": batching, "timestamp": time.time()})


@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    t0 = time.time()
    INFERENCE_REQUESTS_TOTAL.labels(model_version=getattr(model_wrapper, "model_version", "none")).inc()

    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="model not available")

    texts: List[str] = [item.text for item in req.items]

    # Basic safety check remains upstream (not duplicated here). Assume allowed.
    # If batching enabled, enqueue each item and collect results
    try:
        if batcher:
            # enqueue all items concurrently and await results
            coros = [batcher.enqueue(text) for text in texts]
            raw_outputs = await asyncio.gather(*coros)
        else:
            # synchronous batch predict (model_wrapper.predict accepts list)
            raw_outputs = model_wrapper.predict(texts)
    except Exception as e:
        logger.exception("Inference failure: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed") from e

    predictions = []
    for item, out in zip(req.items, raw_outputs):
        predictions.append(Prediction(id=item.id, output=out))

    INFERENCE_PREDICTIONS_TOTAL.labels(model_version=getattr(model_wrapper, "model_version", "none")).inc(len(predictions))
    INFERENCE_LATENCY_HISTOGRAM.observe(time.time() - t0)

    resp = InferenceResponse(flagged=False, safety_reason=None, model_version=getattr(model_wrapper, "model_version", "none"), predictions=predictions)
    return JSONResponse(status_code=200, content=resp.dict())

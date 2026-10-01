183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
"""
    """
    while True:
        try:
            qsize = batcher._queue.qsize() if batcher else 0
            # Observability gauge is per-process
            AEGIS_INFERENCE_QUEUE_DEPTH.set(qsize)
        except Exception:
            logger.exception("Failed to update queue-depth gauge")
        await asyncio.sleep(QUEUE_GAUGE_POLL_INTERVAL)


@app.get("/health")
def health():
    """
    Return structured health including model load, registry info and batch queue size.
    Also set the queue-depth gauge synchronously to avoid scrape race.
    """
    registry_health = registry.health()
    model_info = {"loaded": bool(model_wrapper and getattr(model_wrapper, "model", None) is not None), "model_version": getattr(model_wrapper, "model_version", None) if model_wrapper else None}
    qsize = batcher._queue.qsize() if batcher else 0
    # update gauge immediately
    try:
        AEGIS_INFERENCE_QUEUE_DEPTH.set(qsize)
    except Exception:
        pass

    batching = {"enabled": bool(batcher), "max_batch_size": BATCH_MAX_SIZE, "queue_size": qsize}
    return JSONResponse(status_code=200, content={"ok": True, "registry": registry_health, "model": model_info, "batching": batching, "timestamp": time.time()})


@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    t0 = time.time()
    INFERENCE_REQUESTS_TOTAL.labels(model_version=getattr(model_wrapper, "model_version", "none")).inc()

    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="model not available")

    texts: List[str] = [item.text for item in req.items]

    try:
        if batcher:
            coros = [batcher.enqueue(text) for text in texts]
            raw_outputs = await asyncio.gather(*coros)
        else:
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
aegis_multimodal_ai_system/inference/server_hardened.py

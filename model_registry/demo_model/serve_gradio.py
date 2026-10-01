 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
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
#!/usr/bin/env python3
try:
    import onnxruntime as ort
except Exception:
    ort = None

ORCH_URL = os.environ.get("ORCH_URL", "").rstrip("/")

LOCAL_ONNX = Path("model_registry/example_multimodal/0.1/model.onnx")

def pil_to_rgb_array(img: Image.Image):
    img = img.convert("L").resize((28,28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr[None, None, :, :]  # (1,1,H,W)
    return arr

def call_orchestrator(image: Image.Image, text: str):
    # Prepare payload; base64 image + text
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"image_b64": img_b64, "text": text}
    resp = requests.post(ORCH_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def call_local_onnx(image: Image.Image, text: str):
    if ort is None or not LOCAL_ONNX.exists():
        return {"error": "Local ONNX runtime not available or model not found. Set ORCH_URL or export model."}
    sess = ort.InferenceSession(str(LOCAL_ONNX), providers=["CPUExecutionProvider"])
    img_arr = pil_to_rgb_array(image)
    # text tokens synthetic: map each char to small ints (toy)
    tokens = np.array([[min(99, ord(c) % 100) for c in text[:6].ljust(6)]], dtype=np.int64)
    outs = sess.run(None, {"image": img_arr, "text_tokens": tokens})
    logits = outs[0].tolist()
    return {"logits": logits}

def infer(image: Image.Image, text: str):
    try:
        if ORCH_URL:
            result = call_orchestrator(image, text)
            return json.dumps(result, indent=2)
        else:
            return json.dumps(call_local_onnx(image, text), indent=2)
    except Exception as e:
        return {"error": str(e)}

title = "Aegis demo — multimodal inference"
description = "Upload an image and enter a short text. The demo will call the Aegis orchestrator if ORCH_URL is set, otherwise it runs a local ONNX model."

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}\n\n{description}")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Input image")
        text_in = gr.Textbox(lines=1, placeholder="Enter short text (6 chars used)", label="Text input")
    out = gr.Code(label="Inference result (JSON)")
    run_btn = gr.Button("Run inference")
    run_btn.click(fn=infer, inputs=[img_in, text_in], outputs=out)

if __name__ == "__main__":
    demo.launch()
model_registry/demo_model/serve_gradio.py

#!/usr/bin/env python3
"""
Image encoder pipeline.

- Extracts image embeddings using CLIP (if available) or a lightweight ResNet + pool fallback.
- Emits embeddings as numpy arrays and writes JSON metadata.
- Intended to be used in Argo jobs to process image datasets and call index_to_milvus.py.

Usage:
  python ops/multimodal/image_encoder.py --input /data/images --out /artifacts --model clip
"""
import os
import argparse
import json
import numpy as np
from PIL import Image

try:
    import torch
    from torchvision import transforms
    from torchvision.transforms import functional as F
    # Try CLIP via transformers or openai/clip
    try:
        from transformers import CLIPProcessor, CLIPModel
        _HAS_CLIP = True
    except Exception:
        try:
            import clip
            _HAS_CLIP = True
        except Exception:
            _HAS_CLIP = False
except Exception:
    torch = None
    transforms = None
    _HAS_CLIP = False

def load_image(path):
    return Image.open(path).convert("RGB")

def embed_with_clip(image_paths, out_dir, device="cpu"):
    # prefer transformers CLIP then openai/clip
    if 'CLIPModel' in globals():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for p in image_paths:
            img = load_image(p)
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs).cpu().numpy()[0]
            save_embedding(p, emb, out_dir)
    elif 'clip' in globals():
        model, preprocess = clip.load("ViT-B/32", device=device)
        for p in image_paths:
            img = load_image(p)
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img_t).cpu().numpy()[0]
            save_embedding(p, emb, out_dir)
    else:
        raise RuntimeError("No CLIP available")

def embed_with_resnet(image_paths, out_dir):
    # fallback: simple torchvision ResNet50 pooled features
    from torchvision import models
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    for p in image_paths:
        img = load_image(p)
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = model(t).cpu().numpy()[0]
        save_embedding(p, emb, out_dir)

def save_embedding(path, emb, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    np.save(os.path.join(out_dir, f"{name}.npy"), emb.astype(np.float32))
    meta = {"source": path, "embedding_file": f"{name}.npy", "modality": "image"}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(meta, fh)

def find_images(input_dir):
    exts = (".jpg",".jpeg",".png",".bmp",".tiff")
    out=[]
    for root,_,files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root,f))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="/artifacts")
    p.add_argument("--model", choices=["clip","resnet"], default="clip")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    imgs = find_images(args.input)
    if not imgs:
        print("no images found")
        return
    if args.model == "clip":
        try:
            embed_with_clip(imgs, args.out, device=args.device)
        except Exception as e:
            print("clip failed, fallback to resnet:", e)
            embed_with_resnet(imgs, args.out)
    else:
        embed_with_resnet(imgs, args.out)
    print("done")

if __name__=="__main__":
    main()

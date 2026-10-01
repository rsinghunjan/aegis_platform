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
#!/usr/bin/env python3

def synthetic_caption_for_label(label:int):
    # simple mapping for CIFAR labels
    words = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    caption = words[label]
    # pad to 6 tokens by repeating
    tokens = [caption] * 6
    return tokens

def tokens_to_ids(tokens, vocab_size=100):
    # toy deterministic mapping: hash token -> small int
    ids = [hash(t) % vocab_size for t in tokens]
    return ids

def export_onnx(model, out_path: Path):
    model.eval()
    dummy_img = torch.randn(1,3,32,32)
    dummy_text = torch.randint(0,100,(1,6), dtype=torch.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_img, dummy_text),
        str(out_path),
        input_names=["image", "text_tokens"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={"image": {0: "batch"}, "text_tokens": {0: "batch", 1: "seq"}}
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    model = TinyMultiModal(vocab_size=100, text_emb=32)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training on CIFAR10 (toy captions)...")
    for epoch in range(args.epochs):
        for images, labels in loader:
            # generate token ids per label
            batch_tokens = [tokens_to_ids(synthetic_caption_for_label(int(l))) for l in labels]
            texts = torch.tensor(batch_tokens, dtype=torch.int64)
            logits = model(images, texts)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("Exporting to ONNX...")
    export_onnx(model, out_dir / "model.onnx")
    build_vocab(out_dir, size=100)
    write_metadata(out_dir)
    print("Wrote ONNX model and metadata to", out_dir)

if __name__ == "__main__":
    main()
training/full_pipeline/train_multimodal_cifar.py

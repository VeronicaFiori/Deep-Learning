import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, argparse, yaml
from tqdm import tqdm
from PIL import Image

import torch
import nltk
from ultralytics import YOLO

from src.data import Vocab, build_transforms
from src.model import Captioner
from src.utils import load_checkpoint
from src.decode import beam_search

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

def extract_nouns(sentence: str):
    tokens = nltk.word_tokenize(sentence.lower())
    tags = nltk.pos_tag(tokens)
    return {w for w, t in tags if t.startswith("NN")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    save_dir = cfg["train"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    vocab = Vocab.from_json(os.path.join(save_dir, "vocab.json"))
    ckpt_path = args.ckpt or os.path.join(save_dir, "best.pt")

    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)
    model.eval()

    detector = YOLO("yolov8n.pt")

    root = cfg["data"]["root"]
    images_dir = os.path.join(root, cfg["data"]["images_dir"])
    test_list = os.path.join(root, cfg["data"]["test_list"])

    with open(test_list, "r", encoding="utf-8") as f:
        img_ids = [l.strip() for l in f if l.strip()][:args.max_samples]

    tf = build_transforms(train=False)

    out_path = os.path.join(eval_dir, "hallucination_yolo.jsonl")
    total_prec = 0.0
    n = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for img_id in tqdm(img_ids, desc="hallucination_yolo"):
            path = os.path.join(images_dir, img_id)
            image_pil = Image.open(path).convert("RGB")

            img_tensor = tf(image_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                seq = beam_search(
                    model, img_tensor.squeeze(0), vocab.bos_id, vocab.eos_id, vocab.pad_id,
                    beam_size=args.beam, max_len=cfg["data"]["max_len"], device=device
                )
            caption = vocab.decode(seq)

            nouns = extract_nouns(caption)

            det = detector(image_pil, verbose=False)[0]
            detected = {detector.names[int(c)] for c in det.boxes.cls.tolist()} if det.boxes is not None else set()

            precision = (len(nouns & detected) / len(nouns)) if nouns else 1.0
            total_prec += precision
            n += 1

            w.write(json.dumps({
                "image": img_id,
                "caption": caption,
                "mentioned_nouns": sorted(list(nouns)),
                "yolo_detected": sorted(list(detected)),
                "mention_precision": precision
            }, ensure_ascii=False) + "\n")

    summary = {
        "avg_mention_precision": total_prec / max(n, 1),
        "n_samples": n
    }
    with open(os.path.join(eval_dir, "hallucination_yolo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print(summary)

if __name__ == "__main__":
    main()

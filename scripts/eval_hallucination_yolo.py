# scripts/eval_hallucination_yolo.py
import os, json, argparse, yaml
from tqdm import tqdm
import torch
from ultralytics import YOLO
from PIL import Image
import nltk

from src.model import CaptioningModel
from src.data import Vocab
from src.utils import beam_search

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

def extract_nouns(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    tags = nltk.pos_tag(tokens)
    return {w for w, t in tags if t.startswith("NN")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(cfg["device"])

    save_dir = cfg["train"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    vocab = Vocab.from_json(os.path.join(save_dir, "vocab.json"))
    model = CaptioningModel(cfg, vocab).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    detector = YOLO("yolov8n.pt")

    results = []

    images_dir = os.path.join(cfg["data"]["root"], cfg["data"]["images_dir"])
    test_list = os.path.join(cfg["data"]["root"], cfg["data"]["test_list"])

    with open(test_list) as f:
        img_ids = [l.strip() for l in f]

    for img_id in tqdm(img_ids[:args.max_samples]):
        path = os.path.join(images_dir, img_id)
        image = Image.open(path).convert("RGB")

        with torch.no_grad():
            img_tensor = cfg["data"]["transform"](image).unsqueeze(0).to(device)
            pred_ids = beam_search(model, img_tensor, vocab, beam_size=args.beam)
        caption = vocab.decode(pred_ids)

        nouns = extract_nouns(caption)
        det = detector(image, verbose=False)[0]
        detected = {detector.names[int(c)] for c in det.boxes.cls.tolist()}

        if nouns:
            precision = len(nouns & detected) / len(nouns)
        else:
            precision = 1.0

        results.append({
            "image": img_id,
            "caption": caption,
            "mentioned_objects": list(nouns),
            "detected_objects": list(detected),
            "mention_precision": precision
        })

    with open(os.path.join(eval_dir, "hallucination_yolo.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Saved hallucination analysis.")

if __name__ == "__main__":
    main()

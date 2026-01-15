# scripts/eval_full.py
import os, json, argparse, yaml
from tqdm import tqdm
from collections import defaultdict

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from src.model import CaptioningModel
from src.data import Vocab, Flickr8kCachedDataset
from src.utils import beam_search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device(cfg["device"])

    save_dir = cfg["train"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    vocab = Vocab.from_json(os.path.join(save_dir, "vocab.json"))

    dataset = Flickr8kCachedDataset(
        cfg, split="test", vocab=vocab
    )

    model = CaptioningModel(cfg, vocab).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    refs = defaultdict(list)
    hyps = {}

    for i, sample in enumerate(tqdm(dataset)):
        if args.max_samples and i >= args.max_samples:
            break

        image, _, img_id, captions = sample
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_ids = beam_search(model, image, vocab, beam_size=args.beam)
        pred = vocab.decode(pred_ids)

        hyps[img_id] = [pred]
        for c in captions:
            refs[img_id].append(c)

    scorers = [
        (Bleu(4), ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    results = {}
    for scorer, name in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(name, list):
            for n, s in zip(name, score):
                results[n] = s
        else:
            results[name] = score

    with open(os.path.join(eval_dir, "metrics_coco.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Saved metrics to", eval_dir)
    print(results)

if __name__ == "__main__":
    main()

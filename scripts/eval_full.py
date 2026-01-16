import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, argparse, yaml
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from src.data import Vocab, build_transforms, parse_flickr8k_captions, Flickr8kCachedDataset, collate_fn
from src.model import Captioner
from src.utils import load_checkpoint
from src.decode import beam_search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    root = cfg["data"]["root"]
    prep = os.path.join(root, "prepared")
    save_dir = cfg["train"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    vocab = Vocab.from_json(os.path.join(save_dir, "vocab.json"))
    splits = json.load(open(os.path.join(prep, "splits.json"), "r", encoding="utf-8"))
    encoded = torch.load(os.path.join(prep, "captions_encoded.pt"), map_location="cpu")["encoded"]

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

    tf = build_transforms(train=False)
    test_ds = Flickr8kCachedDataset(root, cfg["data"]["images_dir"], splits["test"], encoded, tf, sample_caption=False)
    #test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
     #                        collate_fn=lambda b: collate_fn(b, vocab.pad_id))

    max_len = cfg["data"]["max_len"]

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id, vocab.bos_id, max_len)
    )


    caps_map = parse_flickr8k_captions(os.path.join(root, cfg["data"]["captions_file"]))

    refs = defaultdict(list)
    hyps = {}

    for i, batch in enumerate(tqdm(test_loader, desc="eval_full")):
        if args.max_samples is not None and i >= args.max_samples:
            break

        image = batch["image"].squeeze(0).to(device)
        image_id = batch["image_id"][0]

        with torch.no_grad():
            seq = beam_search(
                model, image, vocab.bos_id, vocab.eos_id, vocab.pad_id,
                beam_size=cfg["train"]["beam_size_eval"],
                max_len=cfg["data"]["max_len"],
                device=device
            )
        pred = vocab.decode(seq)

        hyps[image_id] = [pred]
        for c in caps_map[image_id]:
            refs[image_id].append(c)

    scorers = [
        (Bleu(4), ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    results = {}
    for scorer, name in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(name, list):
            for n, s in zip(name, score):
                results[n] = float(s)
        else:
            results[name] = float(score)

    out_path = os.path.join(eval_dir, "metrics_full.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print(results)

if __name__ == "__main__":
    main()

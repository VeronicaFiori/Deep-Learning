import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.data import Vocab, build_transforms, parse_flickr8k_captions, Flickr8kCachedDataset, collate_fn
from src.model import Captioner
from src.utils import load_checkpoint
from src.decode import beam_search

def ref_tokenize(s: str):
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.split()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    root = cfg["data"]["root"]
    prep = os.path.join(root, "prepared")

    vocab = Vocab.from_json(os.path.join(prep, "vocab.json"))
    splits = json.load(open(os.path.join(prep, "splits.json"), "r", encoding="utf-8"))
    encoded = torch.load(os.path.join(prep, "captions_encoded.pt"), map_location="cpu")["encoded"]

    ckpt_path = args.ckpt or os.path.join(cfg["train"]["save_dir"], "best.pt")

    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)

    tf = build_transforms(train=False)
    test_ds = Flickr8kCachedDataset(root, cfg["data"]["images_dir"], splits["test"], encoded, tf, sample_caption=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=lambda b: collate_fn(b, vocab.pad_id))

    caps_map = parse_flickr8k_captions(os.path.join(root, cfg["data"]["captions_file"]))
    refs, hyps = [], []
    smooth = SmoothingFunction().method4

    for batch in tqdm(test_loader, desc="eval"):
        image = batch["image"].squeeze(0)
        image_id = batch["image_id"][0]

        seq = beam_search(model, image, vocab.bos_id, vocab.eos_id, vocab.pad_id,
                          beam_size=cfg["train"]["beam_size_eval"], max_len=cfg["data"]["max_len"], device=device)
        hyp = vocab.decode(seq).split()

        ref_caps = caps_map[image_id]
        ref_toks = [ref_tokenize(c) for c in ref_caps]
        refs.append(ref_toks)
        hyps.append(hyp)

    bleu4 = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    print("BLEU-4:", bleu4)

if __name__ == "__main__":
    main()

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import torch
from PIL import Image

from src.data import Vocab, build_transforms
from src.model import Captioner
from src.utils import load_checkpoint
from src.decode import greedy_decode, beam_search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    vocab_path = os.path.join(cfg["train"]["save_dir"], "vocab.json")
    vocab = Vocab.from_json(vocab_path)

    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(args.ckpt, model, optimizer=None, map_location=device)

    tf = build_transforms(train=False)
    img = Image.open(args.image).convert("RGB")
    x = tf(img)

    g = greedy_decode(model, x, vocab.bos_id, vocab.eos_id, max_len=cfg["data"]["max_len"], device=device)
    b = beam_search(model, x, vocab.bos_id, vocab.eos_id, vocab.pad_id,
                    beam_size=args.beam, max_len=cfg["data"]["max_len"], device=device)

    print("Greedy:", vocab.decode(g))
    print("Beam:", vocab.decode(b))

if __name__ == "__main__":
    main()

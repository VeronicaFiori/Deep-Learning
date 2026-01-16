#import sys, os

#from src import model
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import json
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, save_checkpoint
from src.data import Vocab, build_transforms, Flickr8kCachedDataset, collate_fn
from src.model import Captioner

def load_prepared(root: str):
    prep = os.path.join(root, "prepared")
    vocab = Vocab.from_json(os.path.join(prep, "vocab.json"))
    splits = json.load(open(os.path.join(prep, "splits.json"), "r", encoding="utf-8"))
    encoded = torch.load(os.path.join(prep, "captions_encoded.pt"), map_location="cpu")["encoded"]
    return vocab, splits, encoded

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        caps = batch["caption_ids"].to(device)
        #logits = model(images, caps)
        lengths = batch["caption_len"].to(device)
        logits = model(images, caps, lengths=lengths)

        targets = caps[:, 1:]
        targets = targets[:, :logits.size(1)]

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total += float(loss.item()) * images.size(0)
        n += images.size(0)
    return total / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg["seed"])

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    root = cfg["data"]["root"]
    vocab, splits, encoded = load_prepared(root)

    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    train_ds = Flickr8kCachedDataset(root, cfg["data"]["images_dir"], splits["train"], encoded, train_tf, sample_caption=True)
    val_ds = Flickr8kCachedDataset(root, cfg["data"]["images_dir"], splits["val"], encoded, val_tf, sample_caption=False)

    max_len = cfg["data"]["max_len"]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=lambda b: collate_fn(b, vocab.pad_id, vocab.bos_id, max_len),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=lambda b: collate_fn(b, vocab.pad_id, vocab.bos_id, max_len),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg["train"]["num_workers"] > 0),
    )


    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    #save_dir = cfg["train"]["save_dir"]
    #ensure_dir(save_dir)
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # quando hai vocab:
    vocab.to_json(os.path.join(save_dir, "vocab.json"))


    best_val = 1e9

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch} [train]")
        running = 0.0

        for step, batch in enumerate(pbar, 1):
            images = batch["image"].to(device)
            caps = batch["caption_ids"].to(device)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                #logits = model(images, caps)
                lengths = batch["caption_len"].to(device)
                logits = model(images, caps, lengths=lengths)
                targets = caps[:, 1:]
                targets = targets[:, :logits.size(1)]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            if step % cfg["train"]["print_every"] == 0:
                pbar.set_postfix(loss=running / cfg["train"]["print_every"])
                running = 0.0

        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch} [val] loss={val_loss:.4f}")

        save_checkpoint(save_dir, "last.pt", model, optim, epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(save_dir, "best.pt", model, optim, epoch, best_val)
            print("saved best.pt")

    vocab.to_json(os.path.join(save_dir, "vocab.json"))
    print("Done. Best val loss:", best_val)

if __name__ == "__main__":
    main()

#import sys, os

#from src import model
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
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
        fine_tune_encoder=False, #cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    decoder_params = list(model.decoder.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    
    #optim = torch.optim.AdamW(
    #    filter(lambda p: p.requires_grad, model.parameters()),
    #    lr=cfg["train"]["lr"],
    #    weight_decay=cfg["train"]["weight_decay"],
    #)
    optim = torch.optim.AdamW(
        [
        {"params": decoder_params, "lr": cfg["train"]["lr"]},
        {"params": encoder_params, "lr": cfg["train"].get("lr_encoder", 1e-5)},
        ],
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
"""










"""

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

from src.utils import set_seed, save_checkpoint
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
        lengths = batch["caption_len"].to(device)

        logits = model(images, caps, lengths=lengths)

        targets = caps[:, 1:]
        targets = targets[:, :logits.size(1)]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        total += float(loss.item()) * images.size(0)
        n += images.size(0)

    return total / max(n, 1)


def make_optimizer(model, lr_dec, lr_enc, weight_decay):
    
    #Tutorial-style:
    #  - decoder lr: higher
    #  - encoder lr: lower (and only for params with requires_grad=True)
    
    dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]

    param_groups = [{"params": dec_params, "lr": lr_dec}]
    if len(enc_params) > 0:
        param_groups.append({"params": enc_params, "lr": lr_enc})

    optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    # ----- data -----
    root = cfg["data"]["root"]
    vocab, splits, encoded = load_prepared(root)

    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    train_ds = Flickr8kCachedDataset(
        root, cfg["data"]["images_dir"], splits["train"], encoded, train_tf, sample_caption=True
    )
    val_ds = Flickr8kCachedDataset(
        root, cfg["data"]["images_dir"], splits["val"], encoded, val_tf, sample_caption=False
    )

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

    # ----- model -----
    # Tutorial-like: start with encoder frozen, then unfreeze later
    unfreeze_epoch = int(cfg["train"].get("unfreeze_encoder_epoch", 3))  # e.g. 3
    lr_dec = float(cfg["train"]["lr"])                                  # e.g. 3e-4
    lr_enc = float(cfg["train"].get("lr_encoder", 1e-5))                # e.g. 1e-5

    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=False,  # start frozen
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    optim = make_optimizer(
        model,
        lr_dec=lr_dec,
        lr_enc=lr_enc,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2
    )

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    print_every = int(cfg["train"].get("print_every", 50))

    # ----- save dir -----
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    vocab.to_json(os.path.join(save_dir, "vocab.json"))

    best_val = 1e9

    # ----- training -----
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):

        # Unfreeze encoder (tutorial-like) at chosen epoch
        if epoch == unfreeze_epoch:
            print(f">> Unfreezing encoder at epoch {epoch} (lr_encoder={lr_enc})")
            # Rebuild model with fine_tune_encoder=True and load weights
            # (safer than trying to change requires_grad inside)
            state = model.state_dict()

            model = Captioner(
                vocab_size=len(vocab.itos),
                fine_tune_encoder=True,  # now allow finetune
                embed_dim=cfg["model"]["embed_dim"],
                hidden_dim=cfg["model"]["hidden_dim"],
                attn_dim=cfg["model"]["attn_dim"],
                dropout=cfg["model"]["dropout"],
            ).to(device)
            model.load_state_dict(state, strict=True)

            optim = make_optimizer(
                model,
                lr_dec=lr_dec,
                lr_enc=lr_enc,
                weight_decay=float(cfg["train"]["weight_decay"]),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.5, patience=2
            )

        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch} [train]")
        running = 0.0

        for step, batch in enumerate(pbar, 1):
            images = batch["image"].to(device)
            caps = batch["caption_ids"].to(device)
            lengths = batch["caption_len"].to(device)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images, caps, lengths=lengths)
                targets = caps[:, 1:]
                targets = targets[:, :logits.size(1)]
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())

            if step % print_every == 0:
                pbar.set_postfix(loss=running / print_every, lr=optim.param_groups[0]["lr"])
                running = 0.0

        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch} [val] loss={val_loss:.4f}")

        scheduler.step(val_loss)

        save_checkpoint(save_dir, "last.pt", model, optim, epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(save_dir, "best.pt", model, optim, epoch, best_val)
            print("saved best.pt")

    vocab.to_json(os.path.join(save_dir, "vocab.json"))
    print("Done. Best val loss:", best_val)


if __name__ == "__main__":
    main()
"""




import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import yaml
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from src.data import Vocab, build_transforms, Flickr8kCachedDataset, collate_fn
from src.model import Captioner
from src.utils import set_seed, save_checkpoint


def load_prepared(root: str):
    prep = os.path.join(root, "prepared")
    vocab = Vocab.from_json(os.path.join(prep, "vocab.json"))
    splits = json.load(open(os.path.join(prep, "splits.json"), "r", encoding="utf-8"))
    encoded = torch.load(os.path.join(prep, "captions_encoded.pt"), map_location="cpu")["encoded"]
    return vocab, splits, encoded


def sort_batch(images, caps, caplens):
    # caplens: (B,) long
    caplens_sorted, sort_ind = caplens.sort(dim=0, descending=True)
    images = images[sort_ind]
    caps = caps[sort_ind]
    return images, caps, caplens_sorted, sort_ind


@torch.no_grad()
def validate(model, loader, criterion, device, alpha_c=0.0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        images = batch["image"].to(device)
        caps = batch["caption_ids"].to(device)
        caplens = batch["caption_len"].to(device)

        images, caps, caplens, _ = sort_batch(images, caps, caplens)

        # forward (supporta sia return logits che (logits, alphas))
        out = model(images, caps, lengths=caplens)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            scores, alphas = out
        else:
            scores, alphas = out, None

        # decode lengths = caplens - 1 (perché target è shiftato)
        decode_lengths = (caplens - 1).tolist()

        targets = caps[:, 1:]  # (B, L-1)

        # pack to remove pads / extra steps
        scores_packed, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True)
        targets_packed, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True)

        loss = criterion(scores_packed, targets_packed)

        # doubly stochastic attention (opzionale)
        if alpha_c > 0.0 and alphas is not None:
            # alphas: (B, T, N) -> somma su T dovrebbe avvicinarsi a 1 per ogni regione
            loss = loss + alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        total_loss += float(loss.item()) * targets_packed.size(0)
        total_tokens += int(targets_packed.size(0))

    return total_loss / max(total_tokens, 1)


def make_optimizers(model, decoder_lr, encoder_lr, weight_decay, fine_tune_encoder):
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    decoder_opt = torch.optim.AdamW(decoder_params, lr=decoder_lr, weight_decay=weight_decay)

    encoder_opt = None
    if fine_tune_encoder:
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        if len(encoder_params) > 0:
            encoder_opt = torch.optim.AdamW(encoder_params, lr=encoder_lr, weight_decay=weight_decay)

    return encoder_opt, decoder_opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    # ----- data -----
    root = cfg["data"]["root"]
    vocab, splits, encoded = load_prepared(root)

    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    train_ds = Flickr8kCachedDataset(root, cfg["data"]["images_dir"], splits["train"], encoded, train_tf, sample_caption=True)
    val_ds   = Flickr8kCachedDataset(root, cfg["data"]["images_dir"], splits["val"],   encoded, val_tf,   sample_caption=False)

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

    # ----- model -----
    # tutorial-style: puoi partire con encoder frozen e poi sbloccarlo
    unfreeze_epoch = int(cfg["train"].get("unfreeze_encoder_epoch", 3))
    encoder_lr = float(cfg["train"].get("lr_encoder", 1e-5))
    decoder_lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 1e-2))
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    alpha_c = float(cfg["train"].get("alpha_c", 0.0))  # tutorial doubly-stochastic

    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=False,  # start frozen
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id).to(device)

    encoder_opt, decoder_opt = make_optimizers(
        model, decoder_lr=decoder_lr, encoder_lr=encoder_lr, weight_decay=weight_decay, fine_tune_encoder=False
    )

    # AMP
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Save dir
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    vocab.to_json(os.path.join(save_dir, "vocab.json"))

    best_val = 1e9
    epochs_since_improvement = 0
    max_bad_epochs = int(cfg["train"].get("early_stop_patience", 10))
    lr_decay_every = int(cfg["train"].get("lr_decay_every", 4))
    lr_decay_gamma = float(cfg["train"].get("lr_decay_gamma", 0.8))

    def decay_lr(opt):
        if opt is None: 
            return
        for pg in opt.param_groups:
            pg["lr"] *= lr_decay_gamma

    # ----- training loop -----
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):

        # Unfreeze encoder like tutorial
        if epoch == unfreeze_epoch:
            print(f">> Unfreezing encoder at epoch {epoch} (encoder_lr={encoder_lr})")
            # ricrea modello con fine_tune_encoder=True mantenendo pesi
            state = model.state_dict()
            model = Captioner(
                vocab_size=len(vocab.itos),
                fine_tune_encoder=True,
                embed_dim=cfg["model"]["embed_dim"],
                hidden_dim=cfg["model"]["hidden_dim"],
                attn_dim=cfg["model"]["attn_dim"],
                dropout=cfg["model"]["dropout"],
            ).to(device)
            model.load_state_dict(state, strict=True)

            encoder_opt, decoder_opt = make_optimizers(
                model, decoder_lr=decoder_lr, encoder_lr=encoder_lr, weight_decay=weight_decay, fine_tune_encoder=True
            )

        model.train()
        start = time.time()
        running_tokens = 0
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch} [train]")
        for i, batch in enumerate(pbar, 1):
            images = batch["image"].to(device)
            caps = batch["caption_ids"].to(device)
            caplens = batch["caption_len"].to(device)

            # tutorial: sort by caplens desc
            images, caps, caplens, _ = sort_batch(images, caps, caplens)

            encoder_opt and encoder_opt.zero_grad(set_to_none=True)
            decoder_opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(images, caps, lengths=caplens)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    scores, alphas = out
                else:
                    scores, alphas = out, None

                decode_lengths = (caplens - 1).tolist()
                targets = caps[:, 1:]

                scores_packed, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True)
                targets_packed, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True)

                loss = criterion(scores_packed, targets_packed)

                if alpha_c > 0.0 and alphas is not None:
                    loss = loss + alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                if encoder_opt is not None:
                    torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)

            scaler.step(decoder_opt)
            if encoder_opt is not None:
                scaler.step(encoder_opt)
            scaler.update()

            # stats like tutorial: loss per word
            n_tok = int(targets_packed.size(0))
            running_tokens += n_tok
            running_loss += float(loss.item()) * n_tok

            if i % int(cfg["train"].get("print_every", 50)) == 0:
                avg_loss = running_loss / max(running_tokens, 1)
                pbar.set_postfix(loss_per_tok=f"{avg_loss:.4f}", t=f"{(time.time()-start):.1f}s")

        # ----- validation -----
        val_loss = validate(model, val_loader, criterion, device, alpha_c=alpha_c)
        print(f"epoch {epoch} [val] loss_per_tok={val_loss:.4f}")

        # LR decay logic like tutorial (simpler)
        if val_loss < best_val:
            best_val = val_loss
            epochs_since_improvement = 0
            save_checkpoint(save_dir, "best.pt", model, decoder_opt, epoch, best_val)
            print("saved best.pt")
        else:
            epochs_since_improvement += 1
            print(f"epochs since improvement: {epochs_since_improvement}")

            if epochs_since_improvement % lr_decay_every == 0:
                print(">> decaying LR")
                decay_lr(decoder_opt)
                decay_lr(encoder_opt)

            if epochs_since_improvement >= max_bad_epochs:
                print("Early stopping.")
                break

        save_checkpoint(save_dir, "last.pt", model, decoder_opt, epoch, best_val)

    print("Done. Best val loss per token:", best_val)


if __name__ == "__main__":
    main()

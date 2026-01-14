import os
import json
import re
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def simple_tokenize(s: str) -> List[str]:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.split()

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    @classmethod
    def from_json(cls, path: str) -> "Vocab":
        v = json.load(open(path, "r", encoding="utf-8"))
        stoi, itos = v["stoi"], v["itos"]
        return cls(
            stoi=stoi, itos=itos,
            pad_id=stoi["<pad>"], bos_id=stoi["<bos>"], eos_id=stoi["<eos>"], unk_id=stoi["<unk>"]
        )

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, ensure_ascii=False)

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.bos_id, self.pad_id):
                continue
            words.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return " ".join(words)

def build_vocab(captions: List[str], min_freq: int) -> Vocab:
    from collections import Counter
    counter = Counter()
    for c in captions:
        counter.update(simple_tokenize(c))
    words = [w for w, f in counter.items() if f >= min_freq]
    itos = SPECIALS + sorted(words)
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos,
                 pad_id=stoi["<pad>"], bos_id=stoi["<bos>"],
                 eos_id=stoi["<eos>"], unk_id=stoi["<unk>"])

def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def parse_flickr8k_captions(captions_path: str) -> Dict[str, List[str]]:
    caps = {}
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, cap = line.split("\t", 1)
            image_id = left.split("#")[0]
            caps.setdefault(image_id, []).append(cap)
    return caps

def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def encode_caption(vocab: Vocab, caption: str, max_len: int) -> List[int]:
    tokens = simple_tokenize(caption)
    ids = [vocab.bos_id] + [vocab.stoi.get(t, vocab.unk_id) for t in tokens] + [vocab.eos_id]
    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = vocab.eos_id
    return ids

class Flickr8kCachedDataset(Dataset):
    def __init__(self, root: str, images_dir: str, image_ids: List[str],
                 captions_encoded: Dict[str, List[List[int]]],
                 transform, sample_caption: bool = True):
        self.images_dir = os.path.join(root, images_dir)
        self.image_ids = image_ids
        self.captions_encoded = captions_encoded
        self.transform = transform
        self.sample_caption = sample_caption

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        path = os.path.join(self.images_dir, image_id)
        #img = Image.open(path).convert("RGB")
        #img = self.transform(img)
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            # immagine mancante: prova a prendere un altro esempio
            new_idx = (idx + 1) % len(self.image_ids)
            return self.__getitem__(new_idx)
        img = self.transform(img)




        caps = self.captions_encoded[image_id]
        cap_ids = random.choice(caps) if self.sample_caption else caps[0]
        return {"image": img, "cap_ids": cap_ids, "cap_len": len(cap_ids), "image_id": image_id}

def collate_fn(batch, pad_id: int):
    batch = sorted(batch, key=lambda x: x["cap_len"], reverse=True)
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = torch.tensor([b["cap_len"] for b in batch], dtype=torch.long)

    max_len = int(lengths.max().item())
    caps = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        ids = b["cap_ids"]
        caps[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return {"image": images, "caption_ids": caps, "caption_len": lengths, "image_id": [b["image_id"] for b in batch]}

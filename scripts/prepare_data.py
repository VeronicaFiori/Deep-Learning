import os
import json
import argparse
import yaml
import torch

from src.data import parse_flickr8k_captions, read_list, build_vocab, encode_caption

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    root = cfg["data"]["root"]

    caps_map = parse_flickr8k_captions(os.path.join(root, cfg["data"]["captions_file"]))
    train_ids = read_list(os.path.join(root, cfg["data"]["train_list"]))
    val_ids = read_list(os.path.join(root, cfg["data"]["val_list"]))
    test_ids = read_list(os.path.join(root, cfg["data"]["test_list"]))

    train_caps = []
    for img_id in train_ids:
        train_caps.extend(caps_map[img_id])

    vocab = build_vocab(train_caps, min_freq=cfg["data"]["min_word_freq"])

    encoded = {img_id: [encode_caption(vocab, c, cfg["data"]["max_len"]) for c in caps]
               for img_id, caps in caps_map.items()}

    out_dir = os.path.join(root, "prepared")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": vocab.stoi, "itos": vocab.itos}, f, ensure_ascii=False)

    with open(os.path.join(out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, ensure_ascii=False)

    torch.save({"encoded": encoded}, os.path.join(out_dir, "captions_encoded.pt"))

    print("OK. Prepared saved to:", out_dir)
    print("Vocab size:", len(vocab.itos))
    print("Splits:", len(train_ids), len(val_ids), len(test_ids))

if __name__ == "__main__":
    main()

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, argparse, yaml
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from src.data import Vocab, build_transforms
from src.model import Captioner
from src.utils import load_checkpoint
from src.decode import beam_search

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

    cap_model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(ckpt_path, cap_model, optimizer=None, map_location=device)
    cap_model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    judge = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    root = cfg["data"]["root"]
    images_dir = os.path.join(root, cfg["data"]["images_dir"])
    test_list = os.path.join(root, cfg["data"]["test_list"])

    with open(test_list, "r", encoding="utf-8") as f:
        img_ids = [l.strip() for l in f if l.strip()][:args.max_samples]

    tf = build_transforms(train=False)
    out_path = os.path.join(eval_dir, "qwen2vl_judge.jsonl")

    with open(out_path, "w", encoding="utf-8") as w:
        for img_id in tqdm(img_ids, desc="qwen2vl_judge"):
            image_pil = Image.open(os.path.join(images_dir, img_id)).convert("RGB")

            # 1) caption generata dal tuo modello
            img_tensor = tf(image_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                seq = beam_search(
                    cap_model, img_tensor.squeeze(0), vocab.bos_id, vocab.eos_id, vocab.pad_id,
                    beam_size=args.beam, max_len=cfg["data"]["max_len"], device=device
                )
            caption = vocab.decode(seq)

            # 2) judge prompt (chiedi JSON “pulito”)
            prompt = (
                "Sei un valutatore di image captioning.\n"
                "Dato input immagine e caption, valuta fedelta' e allucinazioni.\n"
                "Rispondi SOLO con un JSON valido (niente testo extra) con campi:\n"
                "{faithfulness: 0-5, hallucination: 0-5, coverage: 0-5, notes: string}.\n\n"
                f"CAPTION: {caption}"
            )

            inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(judge.device)
            out = judge.generate(**inputs, max_new_tokens=200)
            judge_text = processor.decode(out[0], skip_special_tokens=True)

            w.write(json.dumps({
                "image": img_id,
                "caption": caption,
                "judge_output": judge_text
            }, ensure_ascii=False) + "\n")

    print("Saved:", out_path)

if __name__ == "__main__":
    main()

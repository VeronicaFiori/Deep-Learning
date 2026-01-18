import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, argparse, yaml
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

import torch

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice  # spesso rompe su Colab (Java)

from src.data import parse_flickr8k_captions

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


def build_prompt(mode: str) -> str:
    # prompt "paper-ready" (deterministico, no creative writing)
    if mode == "brief":
        return "Write a short, accurate caption for this image."
    if mode == "detailed":
        return "Write a detailed, accurate caption for this image. Mention key objects, actions, and scene."
    if mode == "focus_objects":
        return "Describe the image focusing on the main objects and their relationships."
    if mode == "focus_colors":
        return "Describe the image focusing on colors (e.g., clothing, objects, background)."
    if mode == "focus_actions":
        return "Describe the image focusing on actions and what is happening."
    return "Write an accurate caption for this image."


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--prompt_mode", default="neutral",
                    choices=["neutral","brief","detailed","focus_objects","focus_colors","focus_actions"])
    ap.add_argument("--model_id", default="Salesforce/instructblip-flan-t5-xl")
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--dtype", default="fp16", choices=["fp16","fp32"])
    ap.add_argument("--device", default=None, help="override device: cuda/cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    device_str = args.device or cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    root = cfg["data"]["root"]
    img_dir = os.path.join(root, cfg["data"]["images_dir"])

    prep = os.path.join(root, "prepared")
    splits = json.load(open(os.path.join(prep, "splits.json"), "r", encoding="utf-8"))
    test_ids = splits["test"]

    save_dir = cfg["train"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    caps_map = parse_flickr8k_captions(os.path.join(root, cfg["data"]["captions_file"]))

    # --- Load SOTA model ---
    torch_dtype = torch.float16 if (args.dtype == "fp16" and device.type == "cuda") else torch.float32
    processor = InstructBlipProcessor.from_pretrained(args.model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    prompt = build_prompt(args.prompt_mode)

    refs = defaultdict(list)
    hyps = {}

    for i, image_id in enumerate(tqdm(test_ids, desc="eval_full_sota")):
        if args.max_samples is not None and i >= args.max_samples:
            break

        path = os.path.join(img_dir, image_id)
        img = Image.open(path).convert("RGB")

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,          # deterministico per paper
            num_beams=3,              # stabile e spesso migliore del greedy
        )
        pred = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        hyps[image_id] = [pred]
        for c in caps_map[image_id]:
            refs[image_id].append(c)

    # --- Metrics (same as your eval_full.py) ---
    scorers = [
        (Bleu(4), ["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE"),
    ]

    results = {}
    for scorer, name in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(name, list):
            for n, s in zip(name, score):
                results[n] = float(s)
        else:
            results[name] = float(score)

    out_path = os.path.join(eval_dir, f"metrics_full_sota_instructblip_{args.prompt_mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "sota_model_id": args.model_id,
            "prompt_mode": args.prompt_mode,
            "prompt": prompt,
            "max_new_tokens": args.max_new_tokens,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print(results)


if __name__ == "__main__":
    main()

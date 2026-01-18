import os, json, argparse, yaml
from tqdm import tqdm
from PIL import Image

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from src.data import read_list, parse_flickr8k_captions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompt_mode", default="neutral", choices=["neutral","brief","detailed","focus_objects","focus_colors","focus_actions"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    root = cfg["data"]["root"]
    img_dir = os.path.join(root, cfg["data"]["images_dir"])

    # Test ids + refs
    test_list_path = os.path.join(root, cfg["data"]["test_list"]) if "test_list" in cfg["data"] else None
    if test_list_path is None:
        # fallback: se usi prepared/splits.json
        splits = json.load(open(os.path.join(root, "prepared", "splits.json"), "r", encoding="utf-8"))
        test_ids = splits["test"]
    else:
        test_ids = [l.strip() for l in open(test_list_path,"r",encoding="utf-8") if l.strip()]

    caps_path = os.path.join(root, cfg["data"]["captions_file"]) if "captions_file" in cfg["data"] else os.path.join(root, "captions.txt")
    refs = parse_flickr8k_captions(caps_path)  # {image_id: [cap,...]}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SOTA model (InstructBLIP)
    model_id = "Salesforce/instructblip-flan-t5-xl"
    processor = InstructBlipProcessor.from_pretrained(model_id)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    model.eval()

    def build_prompt(mode):
        if mode == "brief":
            return "Write a short caption describing the image."
        if mode == "detailed":
            return "Write a detailed, accurate caption describing the image. Mention key objects, actions, and scene."
        if mode == "focus_objects":
            return "Describe the image focusing on the main objects and their relationships."
        if mode == "focus_colors":
            return "Describe the image focusing on colors and clothing colors."
        if mode == "focus_actions":
            return "Describe the image focusing on actions and what is happening."
        return "Describe the image accurately."

    prompt = build_prompt(args.prompt_mode)

    hyps = {}
    out_path = args.out or os.path.join(cfg["train"]["save_dir"], "eval", f"instructblip_{args.prompt_mode}.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for image_id in tqdm(test_ids, desc="instructblip"):
        path = os.path.join(img_dir, image_id)
        img = Image.open(path).convert("RGB")

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        cap = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        hyps[image_id] = cap

    payload = {
        "prompt_mode": args.prompt_mode,
        "model_id": model_id,
        "hyps": hyps,
        "refs": {k: refs.get(k, []) for k in hyps.keys()}
    }
    json.dump(payload, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()

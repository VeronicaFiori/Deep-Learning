import os, json, argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

STYLES = ["neutral", "technical", "poetic", "funny"]

def prompt_for(style, gt_caption):
    return (
        f"Riscrivi la caption nello stile '{style}'.\n"
        "Regole:\n"
        "- NON aggiungere nuovi oggetti, azioni, colori, luoghi o dettagli.\n"
        "- Mantieni solo fatti già presenti nella caption originale.\n"
        "- Se non sei sicuro di qualcosa, omettilo.\n"
        "Output: una sola frase.\n\n"
        f"CAPTION ORIGINALE: {gt_caption}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--captions_json", required=True)  # mappa {img_id: [cap1..cap5]}
    ap.add_argument("--out_json", default="styled_captions.json")
    ap.add_argument("--max_samples", type=int, default=2000)
    args = ap.parse_args()

    caps = json.load(open(args.captions_json, "r", encoding="utf-8"))

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )

    out = {}
    img_ids = list(caps.keys())[:args.max_samples]

    for img_id in tqdm(img_ids, desc="build_styled"):
        img_path = os.path.join(args.images_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        gt = caps[img_id][0]  # prendi una GT (o random)

        out[img_id] = {}
        for style in STYLES:
            prompt = prompt_for(style, gt)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=80)
            txt = processor.decode(gen[0], skip_special_tokens=True).strip()
            out[img_id][style] = txt

    json.dump(out, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()

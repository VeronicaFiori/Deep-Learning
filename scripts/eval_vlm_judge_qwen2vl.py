# scripts/eval_vlm_judge_qwen2vl.py
import os, json, argparse, yaml
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=False)  # non serve per judge
    ap.add_argument("--max_samples", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    save_dir = cfg["train"]["save_dir"]
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    images_dir = os.path.join(cfg["data"]["root"], cfg["data"]["images_dir"])
    test_list = os.path.join(cfg["data"]["root"], cfg["data"]["test_list"])

    with open(test_list) as f:
        img_ids = [l.strip() for l in f][:args.max_samples]

    results = []

    for img_id in tqdm(img_ids):
        image = Image.open(os.path.join(images_dir, img_id)).convert("RGB")

        prompt = (
            "Valuta la seguente caption rispetto all'immagine.\n"
            "Restituisci SOLO un JSON con:\n"
            "faithfulness (0-5), hallucination (0-5), coverage (0-5), notes.\n"
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        out = model.generate(**inputs, max_new_tokens=200)
        text = processor.decode(out[0], skip_special_tokens=True)

        results.append({
            "image": img_id,
            "judge_output": text
        })

    with open(os.path.join(eval_dir, "qwen2vl_judge.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("Saved Qwen2-VL judge outputs.")

if __name__ == "__main__":
    main()

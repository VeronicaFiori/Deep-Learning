import os, json, argparse, glob
import yaml


METRIC_ORDER = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--eval_dir", default=None, help="override eval dir (default: <save_dir>/eval)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    save_dir = cfg["train"]["save_dir"]
    eval_dir = args.eval_dir or os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    rows = []

    # ---- OURS ----
    ours_path = os.path.join(eval_dir, "metrics_full.json")
    if os.path.exists(ours_path):
        ours_metrics = load_json(ours_path)
        rows.append({
            "model": "Ours (ResNet+Attn+LSTM)",
            "variant": "beam",
            "source": os.path.basename(ours_path),
            "metrics": {k: float(ours_metrics.get(k)) for k in METRIC_ORDER if k in ours_metrics},
        })
    else:
        print("WARNING: missing ours metrics:", ours_path)

    # ---- SOTA (InstructBLIP) ----
    sota_files = sorted(glob.glob(os.path.join(eval_dir, "metrics_full_sota_instructblip_*.json")))
    if not sota_files:
        print("WARNING: no SOTA files found in", eval_dir)

    for p in sota_files:
        data = load_json(p)
        results = data.get("results", {})
        rows.append({
            "model": f"SOTA (InstructBLIP)",
            "variant": data.get("prompt_mode", "unknown"),
            "source": os.path.basename(p),
            "meta": {
                "sota_model_id": data.get("sota_model_id"),
                "prompt": data.get("prompt"),
                "max_new_tokens": data.get("max_new_tokens"),
            },
            "metrics": {k: float(results.get(k)) for k in METRIC_ORDER if k in results},
        })

    # ---- Build Markdown table ----
    header = ["Model", "Variant"] + METRIC_ORDER
    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for r in rows:
        m = r["metrics"]
        md_lines.append(
            "| "
            + " | ".join(
                [r["model"], r["variant"]] + [fmt(m.get(k)) for k in METRIC_ORDER]
            )
            + " |"
        )

    out_json = os.path.join(eval_dir, "results_table.json")
    out_md = os.path.join(eval_dir, "results_table.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "metric_order": METRIC_ORDER}, f, indent=2, ensure_ascii=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("Saved:", out_json)
    print("Saved:", out_md)
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()

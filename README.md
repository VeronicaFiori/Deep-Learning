# Flickr8k Image Captioning (Mid Project)

Baseline: **ResNet50 (pretrained) + Soft Attention + LSTM decoder**.

Features:
- Offline cache of encoded captions (fast data loading)
- Dynamic padding per batch (less wasted compute)
- Greedy + Beam Search decoding
- Streamlit demo

## Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Dataset layout (Flickr8k)
Put files like:
```
data/flickr8k/
  Images/
  Flickr8k.token.txt
  Flickr_8k.trainImages.txt
  Flickr_8k.devImages.txt
  Flickr_8k.testImages.txt
```

## Prepare cache
```bash
python scripts/prepare_data.py --config configs/default.yaml
```

## Train
```bash
python scripts/train.py --config configs/default.yaml
```

## Evaluate (BLEU-4)
```bash
python scripts/eval.py --config configs/default.yaml --ckpt runs/flickr8k_mid/best.pt
```

## Inference
```bash
python scripts/infer.py --config configs/default.yaml --ckpt runs/flickr8k_mid/best.pt --image path/to/img.jpg --beam 5
```

## Streamlit
```bash
streamlit run app/streamlit_app.py
```

### CPU tips
Edit `configs/default.yaml`:
- `device: "cpu"`
- `train.batch_size: 8` (or 16)
- `train.num_workers: 0`

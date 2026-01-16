import os
import yaml
import torch
import streamlit as st
from PIL import Image

from src.data import Vocab, build_transforms
from src.model import Captioner
from src.utils import load_checkpoint
from src.decode import greedy_decode, beam_search

@st.cache_resource
def load_everything():
    cfg = yaml.safe_load(open("configs/default.yaml", "r", encoding="utf-8"))

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    vocab_path = os.path.join(cfg["train"]["save_dir"], "vocab.json")
    ckpt_path = os.path.join(cfg["train"]["save_dir"], "best.pt")

    vocab = Vocab.from_json(vocab_path)

    model = Captioner(
        vocab_size=len(vocab.itos),
        fine_tune_encoder=cfg["model"]["fine_tune_encoder"],
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        attn_dim=cfg["model"]["attn_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    load_checkpoint(ckpt_path, model, optimizer=None, map_location=device)

    tf = build_transforms(train=False)
    return cfg, device, vocab, model, tf

def main():
    st.title("Flickr8k Captioning — ResNet50 + Attention + LSTM")

    st.sidebar.header("Decoding")
    #mode = st.sidebar.selectbox("Mode", ["greedy", "beam"])
    #beam_k = st.sidebar.slider("Beam size", 2, 10, 5)
    mode = st.sidebar.selectbox("Mode", ["greedy", "beam"])
    beam_k = st.sidebar.slider("Beam size", 2, 10, 5)

    detail = st.sidebar.radio("Dettaglio", ["breve", "dettagliato"], index=0)

    if detail == "breve":
        max_len = st.sidebar.slider("Max len", 8, 24, 16)
        alpha = st.sidebar.slider("Length norm alpha", 0.5, 1.5, 1.0)
        min_len = st.sidebar.slider("Min len", 1, 6, 1)
    else:
        max_len = st.sidebar.slider("Max len", 16, 60, 40)
        alpha = st.sidebar.slider("Length norm alpha", 0.0, 1.2, 0.6)
        min_len = st.sidebar.slider("Min len", 1, 12, 6)


    cfg, device, vocab, model, tf = load_everything()

    up = st.file_uploader("Carica un'immagine (jpg/png)", type=["jpg","jpeg","png"])
    if up is None:
        st.info("Allena il modello (best.pt + vocab.json) e poi carica un'immagine.")
        return

    img = Image.open(up).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Genera caption"):
        x = tf(img)
        
        
        if mode == "greedy":
            seq = greedy_decode(model, x, vocab.bos_id, vocab.eos_id, max_len=max_len, device=device, min_len=min_len)
        else:
            seq = beam_search(
                model, x, vocab.bos_id, vocab.eos_id, vocab.pad_id,
                beam_size=beam_k, max_len=max_len, device=device,
                alpha=alpha, min_len=min_len
            )

        
        
        st.success(vocab.decode(seq))

if __name__ == "__main__":
    main()

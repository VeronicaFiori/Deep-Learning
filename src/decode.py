import torch

@torch.no_grad()
def greedy_decode(model, image_tensor, bos_id, eos_id, max_len=40, device="cpu"):
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)
    feats = model.encoder(x)
    h, c = model.decoder.init_state(feats)

    seq = [bos_id]
    cur = torch.tensor([bos_id], device=device)

    for _ in range(max_len - 1):
        emb = model.decoder.embed(cur)
        ctx, _ = model.decoder.attn(feats, h)
        x_in = torch.cat([emb, ctx], dim=1)
        h, c = model.decoder.lstm(x_in, (h, c))
        logits = model.decoder.fc(h)
        nxt = int(torch.argmax(logits, dim=-1).item())
        seq.append(nxt)
        if nxt == eos_id:
            break
        cur = torch.tensor([nxt], device=device)
    return seq

@torch.no_grad()
def beam_search(model, image_tensor, bos_id, eos_id, pad_id, beam_size=5, max_len=40, device="cpu"):
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    feats = model.encoder(x)
    feats = feats.expand(beam_size, feats.size(1), feats.size(2))

    h, c = model.decoder.init_state(feats)

    seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(beam_size, device=device)
    finished = torch.zeros(beam_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        last = seqs[:, -1]
        emb = model.decoder.embed(last)
        ctx, _ = model.decoder.attn(feats, h)
        x_in = torch.cat([emb, ctx], dim=1)

        h, c = model.decoder.lstm(x_in, (h, c))
        logits = model.decoder.fc(h)
        logp = torch.log_softmax(logits, dim=-1)

        logp[finished, :] = -1e9
        logp[finished, pad_id] = 0.0

        total = scores.unsqueeze(1) + logp
        flat = total.reshape(-1)
        top_scores, top_idx = flat.topk(beam_size)

        V = logp.size(1)
        next_beam = top_idx // V
        next_tok = top_idx % V

        seqs = torch.cat([seqs[next_beam], next_tok.unsqueeze(1)], dim=1)
        scores = top_scores

        h = h[next_beam]
        c = c[next_beam]
        feats = feats[next_beam]
        finished = finished[next_beam] | (next_tok == eos_id)

        if finished.all():
            break

    best = int(scores.argmax().item())
    return seqs[best].tolist()

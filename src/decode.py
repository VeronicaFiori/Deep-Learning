import torch

@torch.no_grad()
def greedy_decode(model, image_tensor, bos_id, eos_id, max_len=40, device="cpu", min_len=1):
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)
    feats = model.encoder(x)
    h, c = model.decoder.init_state(feats)

    seq = [bos_id]
    cur = torch.tensor([bos_id], device=device)

    for step in range(max_len - 1):
        emb = model.decoder.embed(cur)
        ctx, _ = model.decoder.attn(feats, h)
        x_in = torch.cat([emb, ctx], dim=1)
        h, c = model.decoder.lstm(x_in, (h, c))
        logits = model.decoder.fc(h)

        if step < min_len:
            logits[:, eos_id] = -1e9

        nxt = int(torch.argmax(logits, dim=-1).item())
        seq.append(nxt)
        if nxt == eos_id:
            break
        cur = torch.tensor([nxt], device=device)

    return seq


@torch.no_grad()
def beam_search(
    model,
    image_tensor,
    bos_id,
    eos_id,
    pad_id,
    beam_size=5,
    max_len=40,
    device="cpu",
    alpha=0.7,          # length normalization strength
    min_len=1,          # minimum generated tokens (excluding BOS)
):
    """
    Returns: best sequence (list[int])
    """
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    feats = model.encoder(x)  # (1, 49, D)
    feats = feats.expand(beam_size, feats.size(1), feats.size(2))  # (B, 49, D)

    h, c = model.decoder.init_state(feats)

    seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)  # (B, 1)
    scores = torch.zeros(beam_size, device=device)  # raw log-prob sum
    finished = torch.zeros(beam_size, dtype=torch.bool, device=device)
    lengths = torch.ones(beam_size, dtype=torch.long, device=device)  # includes BOS as 1

    for step in range(max_len - 1):
        last = seqs[:, -1]  # (B,)

        emb = model.decoder.embed(last)         # (B, E)
        ctx, _ = model.decoder.attn(feats, h)   # (B, C)
        x_in = torch.cat([emb, ctx], dim=1)     # (B, E+C)

        h, c = model.decoder.lstm(x_in, (h, c))     # (B, H)
        logits = model.decoder.fc(h)                # (B, V)
        logp = torch.log_softmax(logits, dim=-1)    # (B, V)

        # Prevent EOS before min_len (excluding BOS):
        if step < min_len:
            logp[:, eos_id] = -1e9

        # Once finished, force PAD and keep score unchanged.
        logp[finished, :] = -1e9
        logp[finished, pad_id] = 0.0

        total = scores.unsqueeze(1) + logp  # (B, V)
        flat = total.reshape(-1)            # (B*V,)

        top_scores, top_idx = flat.topk(beam_size)

        V = logp.size(1)
        next_beam = top_idx // V
        next_tok = top_idx % V

        # Reorder everything
        seqs = torch.cat([seqs[next_beam], next_tok.unsqueeze(1)], dim=1)
        h = h[next_beam]
        c = c[next_beam]
        feats = feats[next_beam]
        finished = finished[next_beam]
        lengths = lengths[next_beam]
        scores = top_scores

        # Update finished/lengths
        just_finished = (~finished) & (next_tok == eos_id)
        finished = finished | just_finished

        # length increases only if not finished before adding token
        lengths = lengths + (~finished).long()  # if it *became* finished now, we won't count further steps anyway

        # Early stop if best possible is already finished and dominates?
        # Simple stop: all finished.
        if finished.all():
            break

    # Length-normalized scores (avoid favoring short captions)
    # lengths currently include BOS; normalize by (length^alpha)
    norm = scores / (lengths.float().pow(alpha))

    best = int(norm.argmax().item())
    return seqs[best].tolist()
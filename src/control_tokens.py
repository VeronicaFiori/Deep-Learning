def prepend_controls(ids, bos_id, style_id=None, focus_id=None, max_len=40):
    # ids contiene già BOS ... EOS ... PAD (tipico)
    out = [bos_id]
    if style_id is not None:
        out.append(style_id)
    if focus_id is not None:
        out.append(focus_id)

    # rimuovi il BOS originale se presente all'inizio
    if len(ids) > 0 and ids[0] == bos_id:
        ids = ids[1:]

    out.extend(ids)

    # taglia a max_len
    out = out[:max_len]
    return out

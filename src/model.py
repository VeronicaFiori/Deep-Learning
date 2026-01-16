import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, fine_tune: bool = False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        for p in self.backbone.parameters():
            p.requires_grad = fine_tune

    def forward(self, images):
        feats = self.backbone(images)  # (B,2048,7,7)
        B, C, H, W = feats.shape
        return feats.view(B, C, H * W).permute(0, 2, 1)  # (B,49,2048)

class SoftAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, feats, hidden):
        f = self.feat_proj(feats)
        h = self.hidden_proj(hidden).unsqueeze(1)
        e = torch.tanh(f + h)
        alpha = torch.softmax(self.score(e).squeeze(-1), dim=1)
        ctx = (feats * alpha.unsqueeze(-1)).sum(dim=1)
        return ctx, alpha

class DecoderLSTMAttn(nn.Module):
    def __init__(self, vocab_size: int, feat_dim=2048, embed_dim=256, hidden_dim=512, attn_dim=512, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = SoftAttention(feat_dim, hidden_dim, attn_dim)
        self.lstm = nn.LSTMCell(embed_dim + feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)

    def init_state(self, feats):
        mean = feats.mean(dim=1)
        h = torch.tanh(self.init_h(mean))
        c = torch.tanh(self.init_c(mean))
        return h, c

    #def forward(self, feats, captions_ids):
    #    B, L = captions_ids.shape
    #    h, c = self.init_state(feats)
    #    emb = self.embed(captions_ids)

    #   logits_steps = []
    #    for t in range(L - 1):
    #        ctx, _ = self.attn(feats, h)
    #        x = torch.cat([emb[:, t, :], ctx], dim=1)
    #        h, c = self.lstm(x, (h, c))
    #        logits_steps.append(self.fc(self.dropout(h)))
    #    return torch.stack(logits_steps, dim=1)

    def forward(self, feats, captions_ids, lengths=None):
        """
        feats: (B, 49, 2048)
        captions_ids: (B, L)  incl. <bos> ... <eos> ... <pad>
        lengths: (B,) lunghezze reali incl. bos/eos (stesso 'cap_len' del collate)
        """
        B, L = captions_ids.shape
        h, c = self.init_state(feats)
        emb = self.embed(captions_ids)

        if lengths is None:
            # fallback: comportamento vecchio
            logits_steps = []
            for t in range(L - 1):
                ctx, _ = self.attn(feats, h)
                x = torch.cat([emb[:, t, :], ctx], dim=1)
                h, c = self.lstm(x, (h, c))
                logits_steps.append(self.fc(self.dropout(h)))
            return torch.stack(logits_steps, dim=1)

        # --- fast path: shrink batch over time ---
        # lengths include BOS+EOS; targets are captions_ids[:, 1:], so usable steps are lengths-1
        max_steps = int(lengths.max().item()) - 1  # steps on which we predict next token
        logits = feats.new_zeros((B, max_steps, self.fc.out_features))

        for t in range(max_steps):
            batch_size_t = int((lengths > (t + 0)).sum().item())
            # t indexes input token position; we predict token at t+1
            ctx, _ = self.attn(feats[:batch_size_t], h[:batch_size_t])
            x = torch.cat([emb[:batch_size_t, t, :], ctx], dim=1)
            h_t, c_t = self.lstm(x, (h[:batch_size_t], c[:batch_size_t]))
            h = torch.cat([h_t, h[batch_size_t:]], dim=0)
            c = torch.cat([c_t, c[batch_size_t:]], dim=0)
            logits[:batch_size_t, t, :] = self.fc(self.dropout(h_t))

        return logits


class Captioner(nn.Module):
    def __init__(self, vocab_size: int, fine_tune_encoder: bool, embed_dim: int, hidden_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.encoder = EncoderCNN(fine_tune=fine_tune_encoder)
        self.decoder = DecoderLSTMAttn(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, attn_dim=attn_dim, dropout=dropout)

    #def forward(self, images, captions_ids):
    #    feats = self.encoder(images)
    #    return self.decoder(feats, captions_ids)
    def forward(self, images, captions_ids, lengths=None):
        feats = self.encoder(images)
        return self.decoder(feats, captions_ids, lengths=lengths)
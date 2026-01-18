"""
import torch
import torch.nn as nn
import torchvision.models as models


#class EncoderCNN(nn.Module):
#    def __init__(self, fine_tune: bool = False):
#        super().__init__()
#        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#        for p in self.backbone.parameters():
#            p.requires_grad = fine_tune

#    def forward(self, images):
#        feats = self.backbone(images)  # (B,2048,7,7)
#        B, C, H, W = feats.shape
#        return feats.view(B, C, H * W).permute(0, 2, 1)  # (B,49,2048)

class EncoderCNN(nn.Module):
    def __init__(self, fine_tune: bool = False, encoded_image_size: int = 14):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # backbone conv fino a layer4 inclusa
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # 14x14 come tutorial (molto meglio di 7x7 per attention)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # di default congela tutto
        for p in self.resnet.parameters():
            p.requires_grad = False

        # fine-tuning solo blocchi alti (come tutorial: children()[5:])
        if fine_tune:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True

    def forward(self, images):
        feats = self.resnet(images)                 # (B,2048,H,W)
        feats = self.adaptive_pool(feats)           # (B,2048,14,14)
        B, C, H, W = feats.shape
        return feats.view(B, C, H * W).permute(0, 2, 1)  # (B,196,2048)
    
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
        self.f_beta = nn.Linear(hidden_dim, feat_dim)
        self.sigmoid = nn.Sigmoid()


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
        
        #feats: (B, 49, 2048)
        #captions_ids: (B, L)  incl. <bos> ... <eos> ... <pad>
        #lengths: (B,) lunghezze reali incl. bos/eos (stesso 'cap_len' del collate)
        
        B, L = captions_ids.shape
        h, c = self.init_state(feats)
        emb = self.embed(captions_ids)

        if lengths is None:
            # fallback: comportamento vecchio
            logits_steps = []
            for t in range(L - 1):
                ctx, _ = self.attn(feats, h)
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))   # (B_t, feat_dim)
                ctx = gate * ctx

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



"""


import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Tutorial-style encoder:
    - ResNet50 conv backbone
    - AdaptiveAvgPool2d to encoded_image_size x encoded_image_size (default 14x14)
    - Fine-tune only high-level blocks when enabled (like tutorial)
    """
    def __init__(self, fine_tune: bool = False, encoded_image_size: int = 14):
        super().__init__()
        #resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Remove avgpool & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        # If fine_tune: unfreeze only higher layers (roughly layer3/layer4)
        if fine_tune:
            for child in list(self.backbone.children())[5:]:
                for p in child.parameters():
                    p.requires_grad = True

        self.encoded_image_size = encoded_image_size

    def forward(self, images):
        feats = self.backbone(images)           # (B,2048,H,W)
        feats = self.adaptive_pool(feats)       # (B,2048,S,S) with S=14
        B, C, H, W = feats.shape
        feats = feats.view(B, C, H * W).permute(0, 2, 1)  # (B, S*S, 2048) => (B,196,2048)
        return feats


class SoftAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, feats, hidden):
        """
        feats: (B, N, D)
        hidden: (B, H)
        """
        f = self.feat_proj(feats)                     # (B,N,A)
        h = self.hidden_proj(hidden).unsqueeze(1)     # (B,1,A)
        e = torch.tanh(f + h)                         # (B,N,A)
        alpha = torch.softmax(self.score(e).squeeze(-1), dim=1)  # (B,N)
        ctx = (feats * alpha.unsqueeze(-1)).sum(dim=1)           # (B,D)
        return ctx, alpha


class DecoderLSTMAttn(nn.Module):
    """
    Tutorial-style decoder:
    - Embedding
    - Soft attention
    - Gating of context: gate = sigmoid(f_beta(h)); ctx = gate * ctx
    - LSTMCell
    - Supports 'lengths' for shrink-batch-over-time
    """
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 2048,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        attn_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = SoftAttention(feat_dim, hidden_dim, attn_dim)

        # Gating (tutorial): f_beta + sigmoid
        self.f_beta = nn.Linear(hidden_dim, feat_dim)
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTMCell(embed_dim + feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)

    def init_state(self, feats):
        mean = feats.mean(dim=1)                 # (B,D)
        h = torch.tanh(self.init_h(mean))        # (B,H)
        c = torch.tanh(self.init_c(mean))        # (B,H)
        return h, c

    def forward(self, feats, captions_ids, lengths=None):
        """
        feats: (B, N, D)
        captions_ids: (B, L) contains <bos> ... <eos> ... <pad>
        lengths: (B,) real lengths incl BOS/EOS
        returns logits: (B, T, V) where T = max_steps = max(lengths)-1
        """
       

        B, L = captions_ids.shape
        N = feats.size(1)  # num regions, 196 se 14x14

        h, c = self.init_state(feats)
        emb = self.embed(captions_ids)  # (B,L,E)

        # fallback: old behavior
        if lengths is None:
            logits_steps = []
            for t in range(L - 1):
                ctx, alpha = self.attn(feats, h)
                alphas[:batch_size_t, t, :] = alpha

                gate = self.sigmoid(self.f_beta(h))
                ctx = gate * ctx
                x = torch.cat([emb[:, t, :], ctx], dim=1)
                h, c = self.lstm(x, (h, c))
                logits_steps.append(self.fc(self.dropout(h)))
            return torch.stack(logits_steps, dim=1)

        # fast path (shrink batch over time)
        #max_steps = int(lengths.max().item()) - 1  # predict next token for steps
        #logits = feats.new_zeros((B, max_steps, self.vocab_size))
        max_steps = int(lengths.max().item()) - 1
        logits = feats.new_zeros((B, max_steps, self.vocab_size))
        alphas = feats.new_zeros((B, max_steps, N))

        for t in range(max_steps):
            batch_size_t = int((lengths > t).sum().item())
            h_t = h[:batch_size_t]
            c_t = c[:batch_size_t]

            #ctx, _ = self.attn(feats[:batch_size_t], h_t)
            ctx, alpha = self.attn(feats[:batch_size_t], h_t)
            alphas[:batch_size_t, t, :] = alpha

            gate = self.sigmoid(self.f_beta(h_t))
            ctx = gate * ctx

            x = torch.cat([emb[:batch_size_t, t, :], ctx], dim=1)
            h_new, c_new = self.lstm(x, (h_t, c_t))
            logits[:batch_size_t, t, :] = self.fc(self.dropout(h_new))

            # put updated states back
            h = torch.cat([h_new, h[batch_size_t:]], dim=0)
            c = torch.cat([c_new, c[batch_size_t:]], dim=0)

        return logits,alphas


class Captioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        fine_tune_encoder: bool,
        embed_dim: int,
        hidden_dim: int,
        attn_dim: int,
        dropout: float,
        encoded_image_size: int = 14,
    ):
        super().__init__()
        self.encoder = EncoderCNN(fine_tune=fine_tune_encoder, encoded_image_size=encoded_image_size)
        self.decoder = DecoderLSTMAttn(
            vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=dropout,
        )

    def forward(self, images, captions_ids, lengths=None):
        feats = self.encoder(images)
        return self.decoder(feats, captions_ids, lengths=lengths)

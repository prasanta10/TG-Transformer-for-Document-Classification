"""TG-Transformer model for document targets (paper Sec 2.3).

Input slice per target document: [target, k helper words]. Each node's input is
    H0 = X[node] + TypeEnc[type] + WLEnc[wl_code]           (300-d)
followed by a Linear(300 -> hidden) projection, LayerNorm and dropout (BERT/
Graph-BERT embedding convention). Then ``num_layers`` BERT-style encoder layers
(paper Eqs. 6-7), each followed by the graph residual "G-Res", and a readout
(paper Eq. 8) with a linear classifier.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def wl_sinusoidal_encoding(codes, d: int) -> torch.Tensor:
    """Fixed sinusoidal encoding of WL codes. Implements paper Eq. 5.

    For dimension index 2l:   sin(code / 10000^(2l / d))
    For dimension index 2l+1: cos(code / 10000^((2l+1) / d))
    (the exponents follow the paper exactly, i.e. the odd index uses 2l+1).

    Args:
        codes: (n,) integer WL codes.
        d: encoding dimension.

    Returns:
        (n, d) float32 tensor.
    """
    codes = torch.as_tensor(np.asarray(codes), dtype=torch.float64)[:, None]
    idx = torch.arange(d, dtype=torch.float64)[None, :]
    angle = codes / torch.pow(10000.0, idx / d)
    enc = torch.where(idx.long() % 2 == 0, torch.sin(angle), torch.cos(angle))
    return enc.float()


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with key padding mask (paper Eqs. 6-7).

    Q, K, V = H W_Q, H W_K, H W_V (Eq. 6); softmax(Q K^T / sqrt(d)) V per head (Eq. 7,
    with d the per-head dimension as in BERT), followed by an output projection.
    """

    def __init__(self, hidden: int, heads: int, attn_dropout: float):
        """Create Q/K/V/output projections."""
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.h, self.dh = heads, hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(attn_dropout)
        self.keep_attn = False      # if True, store the attention probabilities (analysis only)
        self.last_attn = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply attention.

        Args:
            x: (B, L, H) hidden states.
            mask: (B, L) bool, True = real token.
            bias: optional (B, heads, L, L) additive attention bias (N1), added to the
                scaled scores before the mask and the softmax.
        """
        B, L, H = x.shape

        def split(t):
            return t.view(B, L, self.h, self.dh).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        p = torch.softmax(scores, dim=-1)
        if self.keep_attn:
            self.last_attn = p.detach()
        p = self.drop(p)
        ctx = (p @ v).transpose(1, 2).reshape(B, L, H)
        return self.out(ctx)


class EncoderLayer(nn.Module):
    """BERT encoder layer: attention + residual + LayerNorm, FFN + residual + LayerNorm."""

    def __init__(self, hidden: int, heads: int, intermediate: int, dropout: float,
                 attn_dropout: float):
        """Create the attention block and the GELU feed-forward block."""
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden, heads, attn_dropout)
        self.ln1 = nn.LayerNorm(hidden, eps=1e-12)
        self.ff1 = nn.Linear(hidden, intermediate)
        self.ff2 = nn.Linear(intermediate, hidden)
        self.ln2 = nn.LayerNorm(hidden, eps=1e-12)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, H), mask: (B, L) bool, bias: optional (B, heads, L, L)."""
        x = self.ln1(x + self.drop(self.attn(x, mask, bias)))
        x = self.ln2(x + self.drop(self.ff2(F.gelu(self.ff1(x)))))
        return x


class TGTransformer(nn.Module):
    """Text Graph Transformer for document classification (document targets only).

    Args:
        features: (n_nodes, d) node feature table X; becomes an ``nn.Embedding``
            (fine-tuned unless ``freeze_embeddings``). An extra zero row is appended
            as the padding node (gradient always zero via ``padding_idx``).
        node_types: (n_nodes,) 0 = document, 1 = word.
        wl_codes: (n_nodes,) WL codes.
        num_classes: number of labels.
        hidden_size, num_layers, num_heads, intermediate_size, dropout, attn_dropout:
            encoder hyperparameters.
        use_type_enc / use_wl_enc: heterogeneity / WL encodings on or off (E1).
        readout: ``mean`` (masked mean over positions, paper Eq. 8) or ``target``
            (position 0) (E4).
        freeze_embeddings: keep X fixed (E5).
        graph_residual: add the G-Res term at every layer (paper Eq. 7).
        attn_bias: N1 edge-aware attention. ``none`` (baseline) or a bias type
            (``intimacy`` / ``edge``); any value other than ``none`` adds a learnable
            ``attn_beta`` of shape (num_layers, num_heads), initialised to 0, and
            ``forward`` then expects a per-slice bias matrix B.
    """

    def __init__(self, features, node_types, wl_codes, num_classes: int,
                 hidden_size: int = 32, num_layers: int = 2, num_heads: int = 4,
                 intermediate_size: int = 32, dropout: float = 0.5,
                 attn_dropout: float = 0.5, use_type_enc: bool = True,
                 use_wl_enc: bool = True, readout: str = "mean",
                 freeze_embeddings: bool = False, graph_residual: bool = True,
                 attn_bias: str = "none"):
        """Build embeddings, encodings, encoder stack, G-Res projection and classifier."""
        super().__init__()
        if readout not in ("mean", "target"):
            raise ValueError(f"Unknown readout '{readout}'")
        feats = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        n_nodes, d = feats.shape
        self.pad_id = n_nodes
        table = torch.cat([feats, torch.zeros(1, d)], dim=0)
        self.node_emb = nn.Embedding.from_pretrained(table, freeze=freeze_embeddings,
                                                     padding_idx=self.pad_id)
        types = torch.as_tensor(np.asarray(node_types), dtype=torch.long)
        self.register_buffer("node_types", torch.cat([types, torch.ones(1, dtype=torch.long)]),
                             persistent=False)
        self.use_type_enc, self.use_wl_enc = use_type_enc, use_wl_enc
        self.readout, self.graph_residual = readout, graph_residual
        if use_type_enc:
            self.type_emb = nn.Embedding(2, d)
            nn.init.normal_(self.type_emb.weight, std=0.02)
        if use_wl_enc:
            wl = wl_sinusoidal_encoding(wl_codes, d)
            self.register_buffer("wl_table", torch.cat([wl, torch.zeros(1, d)]), persistent=False)
        self.in_proj = nn.Linear(d, hidden_size)
        self.in_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, intermediate_size, dropout, attn_dropout)
            for _ in range(num_layers)])
        if graph_residual:
            # G-Res: one learnable projection of the raw (pre-encoding) features,
            # shared across layers as in Graph-BERT's "raw" graph residual.
            self.res_proj = nn.Linear(d, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        if attn_bias not in ("none", "intimacy", "edge"):
            raise ValueError(f"Unknown attn_bias '{attn_bias}'")
        self.use_attn_bias = attn_bias != "none"
        if self.use_attn_bias:
            # beta[l, h] scales B in head h of layer l; 0 at init = exactly the baseline.
            # Created last and with zeros, so it consumes no random numbers.
            self.attn_beta = nn.Parameter(torch.zeros(num_layers, num_heads))

    def forward(self, node_ids: torch.Tensor, mask: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """Compute class logits.

        Args:
            node_ids: (B, k+1) global node ids; column 0 = target document.
            mask: (B, k+1) bool, True for real nodes.
            bias: (B, k+1, k+1) graph-score matrix B (N1); required iff attn_bias != none.

        Returns:
            (B, num_classes) logits (softmax of paper Eq. 8 is folded into the loss).
        """
        x = self.node_emb(node_ids)                         # raw features X[node]
        h0 = x
        if self.use_type_enc:
            h0 = h0 + self.type_emb(self.node_types[node_ids])
        if self.use_wl_enc:
            h0 = h0 + self.wl_table[node_ids]               # paper Eq. 5
        h = self.drop(self.in_norm(self.in_proj(h0)))       # H(0) projected to hidden size
        res = self.res_proj(x) if self.graph_residual else None
        if self.use_attn_bias != (bias is not None):
            raise ValueError("bias must be given exactly when attn_bias != 'none'")
        for li, layer in enumerate(self.layers):
            lb = None
            if bias is not None:                            # N1: beta[l, h] * B
                lb = self.attn_beta[li][None, :, None, None] * bias[:, None].to(h.dtype)
            h = layer(h, mask, lb)                          # paper Eqs. 6-7
            if res is not None:
                h = h + res                                 # G-Res term of Eq. 7
        if self.readout == "mean":                          # paper Eq. 8: average(H(D))
            m = mask.unsqueeze(-1).to(h.dtype)
            z = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            z = h[:, 0]
        return self.classifier(z)

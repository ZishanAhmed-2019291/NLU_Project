import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (self.h_dim ** -0.5)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2Scratch(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pos_emb_size: int = 128,
        d_model: int = 128,
        n_heads: int = 2,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.0,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size
        self.d_model = d_model
        self.tie_weights = tie_weights

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=not tie_weights)

        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.pos_emb_size:
            raise ValueError(f"Sequence length {seq_len} is larger than pos_emb_size={self.pos_emb_size}.")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.emb_dropout(x)

        mask = self.mask[:, :, :seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, mask)

        return self.lm_head(self.ln_f(x))


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

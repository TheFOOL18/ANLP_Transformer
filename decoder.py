import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rotary_positional_encoding, relative_position_bias, scaled_dot_product_attention
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=512, pos_encoding="rope"):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len
        self.pos_encoding = pos_encoding

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        if self.pos_encoding == "rel_bias":
            self.rel_bias = relative_position_bias(num_heads, max_len=max_len)
        else:
            self.rel_bias = None

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.size()
        B, Lk, _ = k.size()

        q = self.q_proj(q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)

        if self.pos_encoding == "rope":
            q, k = rotary_positional_encoding(q, k, max_seq_len=self.max_len)

        if self.rel_bias is not None:
            bias = self.rel_bias(Lq, Lk)  # (H, Lq, Lk)
        else:
            bias = None

        x, _ = scaled_dot_product_attention(q, k, v, mask=mask, bias=bias)
        x = x.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o_proj(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, max_len=512, pos_encoding="rope"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, max_len=max_len, pos_encoding=pos_encoding)
        # Cross-attention should not use RoPE due to different sequence lengths
        cross_pos_encoding = None if pos_encoding == "rope" else pos_encoding
        self.enc_attn = MultiHeadAttention(d_model, num_heads, max_len=max_len, pos_encoding=cross_pos_encoding)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask, enc_mask):
        # Self-attention
        sa = self.self_attn(x, x, x, mask=self_mask)
        x = self.norm1(x + self.dropout(sa))  # Fix: norm after residual

        # Cross-attention
        ea = self.enc_attn(x, enc_out, enc_out, mask=enc_mask)
        x = self.norm2(x + self.dropout(ea))  # Fix: norm after residual

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))  # Fix: norm after residual
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512, num_layers=6,
                 dropout=0.1, max_len=512, pos_encoding="rope"):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = pos_encoding
        
        # Only use positional embedding for non-RoPE/non-relative bias approaches
        if pos_encoding not in ["rope", "rel_bias"]:
            self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, max_len=max_len, pos_encoding=pos_encoding)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask, enc_mask):
        # Token embeddings
        x = self.embed(x) * math.sqrt(self.d_model)
        
        # Add positional embeddings if not using RoPE or relative bias
        if self.pos_encoding not in ["rope", "rel_bias"]:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, enc_mask)
            
        x = self.norm(x)
        return self.out_proj(x)

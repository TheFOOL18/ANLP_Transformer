import torch
import torch.nn as nn
import math
from utils import apply_rope, get_relative_position_bias

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_len=512):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.max_len = max_len
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Add relative position bias table
        self.relative_bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, nhead)
        )
        
    def get_relative_bias(self, q_len, k_len):
        """Get relative position bias"""
        # Create relative position matrix
        q_coords = torch.arange(q_len, device=self.relative_bias_table.device)[:, None]
        k_coords = torch.arange(k_len, device=self.relative_bias_table.device)[None, :]
        relative_coords = q_coords - k_coords + self.max_len - 1
        relative_coords = relative_coords.clamp(0, 2 * self.max_len - 2)
        
        # Get bias values
        bias = self.relative_bias_table[relative_coords]  # (q_len, k_len, num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, q_len, k_len)
        
    def forward(self, query, key, value, mask=None, pos_encoding=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Apply positional encoding if RoPE
        if pos_encoding == "rope":
            from utils import apply_rope
            Q, K = apply_rope(Q, K)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add relative position bias if specified
        if pos_encoding == "rel_bias":
            rel_bias = self.get_relative_bias(seq_len, key.size(1))
            scores = scores + rel_bias
        
        # Apply mask
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, max_len=512, pos_encoding="rope"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, max_len)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = pos_encoding
        
    def forward(self, x, mask=None):
        # Self-attention with positional encoding type
        attn_output = self.self_attn(x, x, x, mask, self.pos_encoding)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, 
                 dropout=0.1, max_len=512, pos_encoding="rope"):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = pos_encoding
        
        # Only use positional embedding for standard positional encoding
        if pos_encoding not in ["rope", "rel_bias"]:
            self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout, max_len, pos_encoding)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Token embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional embeddings if not using RoPE or relative bias
        if self.pos_encoding not in ["rope", "rel_bias"]:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
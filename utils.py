import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
import heapq

class SimpleVocab:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
    def build_from_sentences(self, sentences, max_size=10000):
        # Count tokens
        counter = Counter()
        for sent in sentences:
            tokens = sent.strip().split()
            counter.update(tokens)
        
        # Build vocabulary
        self.token2id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        
        # Add most frequent tokens
        most_common = counter.most_common(max_size - 4)
        for token, _ in most_common:
            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)
        
        # Create reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        
    @property
    def pad_id(self):
        return self.token2id[self.pad_token]
    
    @property
    def unk_id(self):
        return self.token2id[self.unk_token]
    
    @property
    def bos_id(self):
        return self.token2id[self.bos_token]
    
    @property
    def eos_id(self):
        return self.token2id[self.eos_token]
    
    def __len__(self):
        return len(self.token2id)
    
    def encode(self, sentence, add_bos=False, add_eos=False, max_len=None):
        """Enhanced encode method with special token options"""
        tokens = sentence.strip().split()
        ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        if max_len is not None:
            ids = ids[:max_len]
        
        return ids
    
    def decode(self, ids):
        tokens = []
        for id in ids:
            if id == self.pad_id:
                break
            if id == self.eos_id:
                break
            tokens.append(self.id2token.get(id, self.unk_token))
        return ' '.join(tokens)

def apply_rope(q, k):
    """Apply Rotary Position Embedding - Handle different sequence lengths"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_seq_len = q.shape[-2]
    k_seq_len = k.shape[-2]
    dim = q.shape[-1]
    
    # Only apply RoPE if dimensions are even
    if dim % 2 != 0:
        return q, k
    
    # Create position encodings for q and k separately
    def create_rope_embeddings(seq_len, dim, device):
        position = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float) * 
                            -(math.log(10000.0) / dim))
        
        pos_enc = position * div_term
        cos_pos = torch.cos(pos_enc)
        sin_pos = torch.sin(pos_enc)
        
        # Expand dimensions
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        
        # Duplicate to full dimension
        cos_full = torch.cat([cos_pos, cos_pos], dim=-1)
        sin_full = torch.cat([sin_pos, sin_pos], dim=-1)
        
        return cos_full, sin_full
    
    # Create separate embeddings for q and k
    q_cos, q_sin = create_rope_embeddings(q_seq_len, dim, q.device)
    k_cos, k_sin = create_rope_embeddings(k_seq_len, dim, k.device)
    
    # Apply rotation
    q_rot = q * q_cos + rotate_half(q) * q_sin
    k_rot = k * k_cos + rotate_half(k) * k_sin
    
    return q_rot, k_rot

def rotary_positional_encoding(q, k, max_seq_len=512):
    """RoPE implementation - wrapper around apply_rope"""
    return apply_rope(q, k)

def relative_position_bias(num_heads, max_len=512):
    """Create a relative position bias module"""
    class RelativePositionBias(nn.Module):
        def __init__(self, num_heads, max_len):
            super().__init__()
            self.num_heads = num_heads
            self.max_len = max_len
            # Simple learnable bias table
            self.bias_table = nn.Parameter(torch.zeros(2 * max_len - 1, num_heads))
            
        def forward(self, q_len, k_len):
            # Create relative position matrix
            q_coords = torch.arange(q_len)[:, None]
            k_coords = torch.arange(k_len)[None, :]
            relative_coords = q_coords - k_coords + self.max_len - 1
            relative_coords = relative_coords.clamp(0, 2 * self.max_len - 2)
            
            # Get bias values
            bias = self.bias_table[relative_coords]  # (q_len, k_len, num_heads)
            return bias.permute(2, 0, 1)  # (num_heads, q_len, k_len)
    
    return RelativePositionBias(num_heads, max_len)

def scaled_dot_product_attention(q, k, v, mask=None, bias=None):
    """Scaled dot-product attention with optional bias"""
    # q, k, v: (B, H, L, d_k)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if bias is not None:
        scores = scores + bias.unsqueeze(0)  # Add batch dimension
    
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights

def get_relative_position_bias(seq_len, device):
    """Simple relative position bias"""
    positions = torch.arange(seq_len, device=device)
    relative_positions = positions[:, None] - positions[None, :]
    
    # Simple bias based on distance
    bias = -torch.abs(relative_positions).float() * 0.1
    return bias[None, None, :, :]

def make_pad_mask(x, pad_id):
    """Create padding mask - returns boolean tensor"""
    return (x != pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

def make_causal_mask(size, device):
    """Create causal (triangular) mask for decoder - returns boolean tensor"""
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

def combine_masks(pad_mask, causal_mask):
    """Combine padding and causal masks properly"""
    if pad_mask is not None and causal_mask is not None:
        # Ensure both masks have compatible shapes
        # pad_mask: (B, 1, 1, L), causal_mask: (1, 1, L, L)
        combined = pad_mask & causal_mask
        return combined
    elif pad_mask is not None:
        return pad_mask
    elif causal_mask is not None:
        return causal_mask
    else:
        return None

def make_tgt_mask(tgt, pad_id, device):
    """Create target mask (padding + causal)"""
    seq_len = tgt.size(1)
    pad_mask = make_pad_mask(tgt, pad_id)
    causal_mask = make_causal_mask(seq_len, device)
    
    # Combine masks: both must be True for position to be valid
    return pad_mask & causal_mask

def collate_pairs(batch, src_pad_id, tgt_pad_id, max_src_len=128, max_tgt_len=128):
    """Collate function for DataLoader"""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_batch = [seq[:max_src_len] for seq in src_batch]
    tgt_batch = [seq[:max_tgt_len] for seq in tgt_batch]
    
    max_src = max(len(seq) for seq in src_batch)
    max_tgt = max(len(seq) for seq in tgt_batch)
    
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded.append(src + [src_pad_id] * (max_src - len(src)))
        tgt_padded.append(tgt + [tgt_pad_id] * (max_tgt - len(tgt)))
    
    return torch.LongTensor(src_padded), torch.LongTensor(tgt_padded)

# DECODING STRATEGIES

def greedy_decode(model, src, src_mask, vocab, max_len=100, device='cpu'):
    """Greedy decoding strategy"""
    model.eval()
    batch_size = src.size(0)
    
    # Start with BOS token
    tgt = torch.tensor([[vocab.bos_id]] * batch_size, device=device)
    
    for _ in range(max_len):
        tgt_mask = make_causal_mask(tgt.size(1), device)
        
        with torch.no_grad():
            logits = model(src, tgt, src_mask, tgt_mask)
            
        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if all sequences have EOS
        if (next_token == vocab.eos_id).all():
            break
    
    return tgt

def top_k_decode(model, src, src_mask, vocab, k=10, max_len=100, temperature=1.0, device='cpu'):
    """Top-k sampling decode"""
    model.eval()
    batch_size = src.size(0)
    
    # Start with BOS token
    tgt = torch.tensor([[vocab.bos_id]] * batch_size, device=device)
    
    for _ in range(max_len):
        tgt_mask = make_causal_mask(tgt.size(1), device)
        
        with torch.no_grad():
            logits = model(src, tgt, src_mask, tgt_mask)
            logits = logits[:, -1, :] / temperature
            
        # Top-k filtering
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Sample from top-k
        probs = F.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, 1)
        next_token = torch.gather(top_k_indices, -1, sampled_indices)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if all sequences have EOS
        if (next_token == vocab.eos_id).all():
            break
    
    return tgt

class BeamSearchNode:
    def __init__(self, hidden_state, prev_node, token_id, log_prob, length):
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length
        
    def eval_score(self, alpha=0.6):
        # Length normalization
        return self.log_prob / ((self.length + 5) / 6) ** alpha
    
    def __lt__(self, other):
        return self.eval_score() < other.eval_score()

def beam_search_decode(model, src, src_mask, vocab, beam_width=5, max_len=100, device='cpu'):
    """Beam search decoding"""
    model.eval()
    batch_size = src.size(0)
    
    # For simplicity, process one sequence at a time
    results = []
    
    for i in range(batch_size):
        src_single = src[i:i+1]
        src_mask_single = src_mask[i:i+1] if src_mask is not None else None
        
        # Initialize beam
        start_token = torch.tensor([[vocab.bos_id]], device=device)
        start_node = BeamSearchNode(start_token, None, vocab.bos_id, 0.0, 1)
        
        # Priority queue for beam search
        beam = [start_node]
        completed = []
        
        for step in range(max_len):
            if not beam:
                break
                
            candidates = []
            
            for node in beam:
                if node.token_id == vocab.eos_id:
                    completed.append(node)
                    continue
                
                tgt = node.hidden_state
                tgt_mask = make_causal_mask(tgt.size(1), device)
                
                with torch.no_grad():
                    logits = model(src_single, tgt, src_mask_single, tgt_mask)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Get top beam_width tokens
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                
                for j in range(beam_width):
                    token_id = top_indices[0, j].item()
                    log_prob = node.log_prob + top_log_probs[0, j].item()
                    
                    new_tgt = torch.cat([tgt, torch.tensor([[token_id]], device=device)], dim=1)
                    new_node = BeamSearchNode(new_tgt, node, token_id, log_prob, node.length + 1)
                    candidates.append(new_node)
            
            # Select top beam_width candidates
            beam = sorted(candidates, reverse=True)[:beam_width]
        
        # Get best sequence
        if completed:
            best_node = max(completed, key=lambda x: x.eval_score())
        else:
            best_node = max(beam, key=lambda x: x.eval_score()) if beam else start_node
        
        # Reconstruct sequence
        sequence = []
        current = best_node
        while current.prev_node is not None:
            sequence.append(current.token_id)
            current = current.prev_node
        sequence.reverse()
        
        results.append(sequence)
    
    # Convert to tensor format
    max_seq_len = max(len(seq) for seq in results) if results else 1
    padded_results = []
    
    for seq in results:
        padded_seq = seq + [vocab.pad_id] * (max_seq_len - len(seq))
        padded_results.append(padded_seq)
    
    return torch.tensor(padded_results, device=device)

def decode_sequences(model, src, src_mask, vocab, strategy='greedy', **kwargs):
    """Unified decoding interface"""
    if strategy == 'greedy':
        return greedy_decode(model, src, src_mask, vocab, **kwargs)
    elif strategy == 'top_k' or strategy == 'topk':
        return top_k_decode(model, src, src_mask, vocab, **kwargs)
    elif strategy == 'beam_search' or strategy == 'beam':
        return beam_search_decode(model, src, src_mask, vocab, **kwargs)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")

def compute_bleu(references, hypotheses):
    """Compute BLEU score"""
    def get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams
    
    if len(references) != len(hypotheses):
        return 0.0
    
    total_score = 0.0
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()
        
        if len(hyp_tokens) == 0:
            continue
            
        # Calculate precision for n-grams (n=1 to 4)
        precisions = []
        for n in range(1, 5):
            ref_ngrams = get_ngrams(ref_tokens, n)
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
                
            matches = sum(1 for ngram in hyp_ngrams if ngram in ref_ngrams)
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)
        
        # Brevity penalty
        bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
        
        # BLEU score
        if all(p > 0 for p in precisions):
            score = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
        else:
            score = 0.0
            
        total_score += score
    
    return total_score / len(references) * 100 if references else 0.0
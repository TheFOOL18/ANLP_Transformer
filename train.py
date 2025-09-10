import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import csv
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from utils import (
    SimpleVocab, 
    collate_pairs, 
    make_pad_mask, 
    make_causal_mask, 
    make_tgt_mask,
    compute_bleu,
    greedy_decode,
    beam_search_decode, 
    top_k_decode
)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        enc_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        return dec_output

class ParallelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_tsv_pairs(path, src_vocab, tgt_vocab, max_len=128):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line.split('\t')
            # Fix: SimpleVocab.encode() doesn't have add_bos, add_eos parameters
            s_ids = [src_vocab.bos_id] + src_vocab.encode(src) + [src_vocab.eos_id]
            t_ids = [tgt_vocab.bos_id] + tgt_vocab.encode(tgt) + [tgt_vocab.eos_id]
            
            # Truncate if too long
            s_ids = s_ids[:max_len]
            t_ids = t_ids[:max_len]
            
            pairs.append((s_ids, t_ids))
    return pairs


# ---------------- Decoding strategies ----------------
def decode(model, enc_out, src_mask, vocab_tgt, device, max_len=50,
           strategy="greedy", beam_size=5, topk=10):
    """
    Decode with different strategies.
    model: {"encoder": Encoder, "decoder": Decoder}
    enc_out: (B, Ls, d_model)
    src_mask: (B, 1, 1, Ls)
    """
    B = enc_out.size(0)

    # --------- Fast batch-parallel greedy decoding ---------
    if strategy == "greedy":
        ys = torch.full((B, 1), vocab_tgt.bos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_pad_mask(ys, vocab_tgt.pad_id).to(device) & make_causal_mask(ys.size(1), device).to(device)
            logits = model['decoder'](ys, enc_out, self_mask=tgt_mask, enc_mask=src_mask)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        return ys

    # --------- Beam search decoding (per sequence) ---------
    elif strategy == "beam":
        results = []
        for b in range(B):
            beams = [(torch.tensor([vocab_tgt.bos_id], device=device).unsqueeze(0), 0.0)]  # (seq, score)
            for _ in range(max_len - 1):
                new_beams = []
                for seq, score in beams:
                    tgt_mask = make_pad_mask(seq, vocab_tgt.pad_id).to(device) & make_causal_mask(seq.size(1), device).to(device)
                    logits = model['decoder'](seq, enc_out[b:b+1], self_mask=tgt_mask, enc_mask=src_mask[b:b+1])
                    probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                    topk_probs, topk_ids = probs.topk(beam_size)
                    for prob, idx in zip(topk_probs, topk_ids):
                        new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                        new_beams.append((new_seq, score + prob.item()))
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            results.append(beams[0][0])
        return torch.cat(results, dim=0)

    # --------- Top-k sampling decoding (per sequence) ---------
    elif strategy == "topk":
        results = []
        for b in range(B):
            seq = torch.tensor([[vocab_tgt.bos_id]], device=device)
            for _ in range(max_len - 1):
                tgt_mask = make_pad_mask(seq, vocab_tgt.pad_id).to(device) & make_causal_mask(seq.size(1), device).to(device)
                logits = model['decoder'](seq, enc_out[b:b+1], self_mask=tgt_mask, enc_mask=src_mask[b:b+1])
                probs = F.softmax(logits[:, -1, :], dim=-1).squeeze(0)
                topk_probs, topk_ids = probs.topk(topk)
                next_token = topk_ids[torch.multinomial(topk_probs, 1)]
                seq = torch.cat([seq, next_token.view(1, 1)], dim=1)
            results.append(seq)
        return torch.cat(results, dim=0)

    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")



# ---------------- Training loop ----------------
def train_epoch(model, optimizer, criterion, train_loader, device, pad_id, tgt_vocab):
    model.train()  # Now this will work
    total_loss = 0
    
    for src, tgt in tqdm(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        src_mask = make_pad_mask(src, pad_id)
        tgt_mask = make_tgt_mask(tgt_input, pad_id, device)
        
        optimizer.zero_grad()
        
        # Forward pass - now using the unified model
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


# ---------------- Evaluation ----------------
def evaluate(model, val_loader, device, pad_id, tgt_vocab, strategy='greedy', beam_size=5, topk=10):
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            
            # Create source mask
            src_mask = make_pad_mask(src, pad_id)
            
            # Generate predictions using the unified model
            if strategy == 'greedy':
                pred_ids = greedy_decode(model, src, src_mask, tgt_vocab, device=device)
            elif strategy == 'beam':
                pred_ids = beam_search_decode(model, src, src_mask, tgt_vocab, beam_width=beam_size, device=device)
            elif strategy == 'topk':
                pred_ids = top_k_decode(model, src, src_mask, tgt_vocab, k=topk, device=device)
            
            # Convert to text
            for i in range(src.size(0)):
                ref_text = tgt_vocab.decode(tgt[i].tolist())
                hyp_text = tgt_vocab.decode(pred_ids[i].tolist())
                references.append(ref_text)
                hypotheses.append(hyp_text)
    
    return compute_bleu(references, hypotheses)


# ---------------- Main ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building vocabs...")
    src_vocab = SimpleVocab()
    tgt_vocab = SimpleVocab()
    src_sents = [line.strip().split('\t')[0] for line in open(args.train_file, encoding='utf-8')]
    tgt_sents = [line.strip().split('\t')[1] for line in open(args.train_file, encoding='utf-8')]
    src_vocab.build_from_sentences(src_sents, max_size=args.vocab_size)
    tgt_vocab.build_from_sentences(tgt_sents, max_size=args.vocab_size)
    print("Vocab sizes:", len(src_vocab), len(tgt_vocab))

    train_pairs = load_tsv_pairs(args.train_file, src_vocab, tgt_vocab, max_len=args.max_len)
    val_pairs = load_tsv_pairs(args.valid_file, src_vocab, tgt_vocab, max_len=args.max_len)
    train_ds = ParallelDataset(train_pairs)
    val_ds = ParallelDataset(val_pairs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: collate_pairs(x, src_vocab.pad_id, tgt_vocab.pad_id,
                                                                 max_src_len=args.max_len, max_tgt_len=args.max_len))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda x: collate_pairs(x, src_vocab.pad_id, tgt_vocab.pad_id,
                                                               max_src_len=args.max_len, max_tgt_len=args.max_len))

    encoder = Encoder(len(src_vocab), args.d_model, args.nhead, args.d_ff, args.num_layers,
                      dropout=args.dropout, max_len=args.max_len, pos_encoding=args.pos_encoding)
    decoder = Decoder(len(tgt_vocab), args.d_model, args.nhead, args.d_ff, args.num_layers,
                      dropout=args.dropout, max_len=args.max_len, pos_encoding=args.pos_encoding)

    model = Transformer(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    best_bleu = -1.0
    os.makedirs(args.save_dir, exist_ok=True)

    log_rows = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, criterion, train_loader, device, src_vocab.pad_id, tgt_vocab)
        bleu = evaluate(model, val_loader, device, src_vocab.pad_id, tgt_vocab,
                        strategy=args.decoding, beam_size=args.beam_size, topk=args.topk)
        print(f"Epoch {epoch}: loss {loss:.4f}, val BLEU {bleu:.2f}")
        log_rows.append((epoch, loss, bleu))

        ckpt = {
            "model_state_dict": model.state_dict(),  # Save the unified model
            "src_vocab": src_vocab.token2id,
            "tgt_vocab": tgt_vocab.token2id,
            "args": args
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"ckpt_epoch{epoch}.pt"))
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

    with open(os.path.join(args.save_dir, "train_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "bleu"])
        writer.writerows(log_rows)

    print("Training finished. Best BLEU:", best_bleu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--pos_encoding", choices=["rope", "rel_bias"], default="rope")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--decoding", choices=["greedy", "beam", "topk"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    main(args)
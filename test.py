import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm  # <-- added for progress bar

from encoder import Encoder
from decoder import Decoder
from utils import SimpleVocab, collate_pairs, make_pad_mask, make_causal_mask, compute_bleu

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return dec_output
# ---------------- Dataset ----------------
class ParallelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_tsv_pairs(path, src_vocab, tgt_vocab=None, max_len=128, only_src=False):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if only_src:
                src = line
                s_ids = src_vocab.encode(src, add_bos=True, add_eos=True, max_len=max_len)
                pairs.append((s_ids, [tgt_vocab.pad_id] * 2 if tgt_vocab else [0, 0]))
            else:
                src, tgt = line.split('\t')
                s_ids = src_vocab.encode(src, add_bos=True, add_eos=True, max_len=max_len)
                if tgt_vocab is not None:
                    t_ids = tgt_vocab.encode(tgt, add_bos=True, add_eos=True, max_len=max_len)
                else:
                    t_ids = [0]
                pairs.append((s_ids, t_ids))
    return pairs


def rebuild_vocab_from_ckpt(token2id_dict):
    """Rebuild SimpleVocab from saved token2id dictionary"""
    v = SimpleVocab()
    v.token2id = token2id_dict
    v.id2token = {idx: token for token, idx in token2id_dict.items()}
    # Don't set the ID properties - they're computed automatically
    return v


# ---------------- Decoding ----------------
def decode(model, src, src_mask, vocab_tgt, device, max_len=80,
           strategy="greedy", beam_size=5, topk=10):
    """Updated decode function using unified model"""
    model.eval()
    B = src.size(0)
    results = []

    for b in range(B):
        if strategy == "greedy":
            ys = torch.full((1, 1), vocab_tgt.bos_id, dtype=torch.long, device=device)
            for _ in range(max_len - 1):
                tgt_mask = make_pad_mask(ys, vocab_tgt.pad_id).to(device) & make_causal_mask(ys.size(1), device).to(device)
                
                logits = model(src[b:b+1], ys, src_mask[b:b+1], tgt_mask)
                
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == vocab_tgt.eos_id:
                    break
            results.append(ys[0].cpu().tolist())

        elif strategy == "beam":
            from utils import beam_search_decode
            pred = beam_search_decode(model, src[b:b+1], src_mask[b:b+1], vocab_tgt, 
                                    beam_width=beam_size, device=device, max_len=max_len)
            results.append(pred[0].cpu().tolist())

        elif strategy == "topk":
            # Add the missing topk implementation
            ys = torch.full((1, 1), vocab_tgt.bos_id, dtype=torch.long, device=device)
            for _ in range(max_len - 1):
                tgt_mask = make_pad_mask(ys, vocab_tgt.pad_id).to(device) & make_causal_mask(ys.size(1), device).to(device)
                
                logits = model(src[b:b+1], ys, src_mask[b:b+1], tgt_mask)
                logits = logits[:, -1, :] / 1.0  # temperature
                
                # Top-k filtering
                top_k_logits, top_k_indices = torch.topk(logits, topk, dim=-1)
                
                # Sample from top-k
                probs = torch.softmax(top_k_logits, dim=-1)
                sampled_indices = torch.multinomial(probs, 1)
                next_token = torch.gather(top_k_indices, -1, sampled_indices)
                
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == vocab_tgt.eos_id:
                    break
            results.append(ys[0].cpu().tolist())

        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

    return results


# ---------------- Evaluation ----------------
def evaluate_model(ckpt_path, test_file, batch_size=32, only_src=False, out_translations=None,
                   device=None, strategy="greedy", beam_size=5, topk=10):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    src_vocab = rebuild_vocab_from_ckpt(ckpt['src_vocab'])
    tgt_vocab = rebuild_vocab_from_ckpt(ckpt['tgt_vocab'])
    args = ckpt['args']
    
    # Handle both dict and Namespace objects
    if isinstance(args, dict):
        # If args is already a dictionary
        model_args = args
    else:
        # If args is a Namespace, convert to dict
        model_args = vars(args)

    # Create the unified Transformer model
    encoder = Encoder(len(src_vocab), model_args['d_model'], model_args['nhead'], 
                      model_args['d_ff'], model_args['num_layers'],
                      dropout=model_args.get('dropout', 0.1), 
                      max_len=model_args['max_len'], 
                      pos_encoding=model_args['pos_encoding'])
    
    decoder = Decoder(len(tgt_vocab), model_args['d_model'], model_args['nhead'], 
                      model_args['d_ff'], model_args['num_layers'],
                      dropout=model_args.get('dropout', 0.1), 
                      max_len=model_args['max_len'], 
                      pos_encoding=model_args['pos_encoding'])
    
    model = Transformer(encoder, decoder).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    pairs = load_tsv_pairs(test_file, src_vocab, tgt_vocab if not only_src else None,
                           max_len=model_args['max_len'], only_src=only_src)
    ds = ParallelDataset(pairs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate_pairs(x, src_vocab.pad_id, tgt_vocab.pad_id,
                                  max_src_len=128, max_tgt_len=128))

    refs, hyps, out_lines = [], [], []

    with torch.no_grad():
        for idx, (src, tgt) in enumerate(tqdm(loader, desc="Evaluating", unit="batch")):
            src = src.to(device)
            src_mask = make_pad_mask(src, src_vocab.pad_id).to(device)
            enc_out = model.encoder(src, src_mask)

            decoded_ids = decode(model, src, src_mask, tgt_vocab, device,
                                 max_len=tgt.size(1) if not only_src else model_args['max_len'],
                                 strategy=strategy, beam_size=beam_size, topk=topk)

            for i in range(src.size(0)):
                ref = tgt_vocab.decode(tgt[i].cpu().tolist()) if not only_src else ""
                hyp = tgt_vocab.decode(decoded_ids[i])
                refs.append(ref)
                hyps.append(hyp)
                out_lines.append(f"REF: {ref}\nHYP: {hyp}\n")

    bleu = None
    if not only_src:
        bleu = compute_bleu(refs, hyps)
        print("BLEU on test set:", bleu)
    else:
        print("Translation done (no references so BLEU not computed).")

    if out_translations:
        with open(out_translations, 'w', encoding='utf-8') as f:
            f.writelines("\n".join(out_lines))
        print("Wrote translations to:", out_translations)

    return bleu, refs, hyps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only_src", action="store_true")
    parser.add_argument("--out", default="translations.txt")
    parser.add_argument("--decoding", choices=["greedy", "beam", "topk"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    evaluate_model(args.ckpt, args.test_file,
                   batch_size=args.batch_size,
                   only_src=args.only_src,
                   out_translations=args.out,
                   strategy=args.decoding,
                   beam_size=args.beam_size,
                   topk=args.topk)

# Transformer Translation Models

This folder contains pretrained Transformer models trained with **Rotary Position Embedding (RoPE)** and **Relative Bias** for translation tasks.

## Pretrained Models
The pretrained models (`.pt` files) can be downloaded from the following Google Drive link:

🔗 [Download Models](https://drive.google.com/drive/folders/1DFL22z7db16qza10fK8dzP01kYysli9y?usp=sharing)

- **RoPE Model:** `RoPE_best.pt`
- **Relative Bias Model:** `rel_bias_best.pt`

---

## Running the Code

The script `test.py` supports multiple decoding strategies:
- **Greedy Decoding**
- **Beam Search Decoding**
- **Top-K Sampling**

Make sure you have your **test file** (e.g., `test.tsv`) ready before running the commands below.

---

### 🔹 Running RoPE Model

**Greedy Decoding**
```bash
!python test.py --ckpt RoPE_best.pt --test_file test.tsv --decoding greedy --out translations_greedy.txt
```

**Beam Search Decoding**
```bash
!python test.py --ckpt RoPE_best.pt --test_file test.tsv --decoding beam --beam_size 5 --out translations_beam.txt
```

**Top-k Decoding**
```bash
!python test.py --ckpt RoPE_best.pt --test_file test.tsv --decoding topk --topk 10 --out translations_topk.txt
```

### 🔹 Running Relative Bias Model

**Greedy Decoding**
```bash
!python test.py --ckpt rel_bias_best.pt --test_file test.tsv --decoding greedy --out translations_greedy.txt
```

**Beam Search Decoding**
```bash
!python test.py --ckpt rel_bias_best.pt --test_file test.tsv --decoding beam --beam_size 5 --out translations_beam.txt
```

**Top-k Decoding**
```bash
!python test.py --ckpt rel_bias_best.pt --test_file test.tsv --decoding topk --topk 10 --out translations_topk.txt
```

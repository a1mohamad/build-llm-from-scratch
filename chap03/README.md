# Chapter 3 — Coding Attention Mechanisms

This folder contains my study notebooks for Chapter 3 of *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The notebooks are a personal educational implementation. They are not the official companion code of the book.

The goal of the chapter is to build the attention mechanism step by step in PyTorch, starting from a very simple version and ending with the multi-head attention layer used in GPT-style models.

---

## Notebook Order

The notebooks build on each other. Later notebooks copy classes from earlier ones, so study them in this order.

| # | Notebook | What you study | Main result |
|---|----------|----------------|-------------|
| 1 | `attention-mechanism.ipynb` | Simplified attention without trainable weights | Attention scores, attention weights, and context vectors for a 6-token input |
| 2 | `self-attention.ipynb` | Self-attention with trainable query, key, and value weights | `SelfAttentionV1` and `SelfAttentionV2` classes |
| 3 | `casual-attention.ipynb` | Causal masking and dropout on attention weights | `CasualAttention` class that works on batches |
| 4 | `multi-head-attention.ipynb` | Several attention heads working in parallel | `MultiHeadAttentionWrapper` and `MultiHeadAttention` classes |

> **Note on spelling:** the files and classes use the word `casual`. The technique is called *causal* attention. The README keeps the notebook names as they are, so the paths stay correct.

---

## Learning Objectives

After studying this chapter, you should be able to:

- explain how a dot product measures the similarity between two token vectors;
- turn attention scores into attention weights with softmax;
- compute a context vector as a weighted sum of input vectors;
- explain the roles of the query, key, and value projections;
- explain why scores are divided by the square root of the key dimension;
- hide future tokens with a causal mask;
- apply dropout to attention weights;
- explain how multi-head attention splits `d_out` into several heads and joins the results.

---

## What Each Notebook Does

### 1. `attention-mechanism.ipynb` — Simplified Attention

All notebooks use the same small input: six tokens ("Your Journey starts with one step"), each with three numbers (a tensor of shape 6 × 3).

The notebook uses the second token ("Journey") as the query and shows the full path from scores to a context vector:

1. dot products between the query and every token give attention scores;
2. the scores are normalized in three ways: division by the sum, a hand-written `softmax_naive`, and `torch.softmax`;
3. the weights are used to build the context vector of the second token;
4. the same idea is repeated for all tokens with a loop, then with one matrix product (`inputs @ inputs.T`);
5. `softmax(..., dim=-1)` gives weights where each row sums to 1, and `attn_weights @ inputs` gives all context vectors.

Why it matters: this notebook shows that attention is only a weighted average. Everything after this notebook adds learnable parts on top of this idea.

What the stored outputs show: the context vector for the second token from the loop matches the second row of the matrix result (`[0.4419, 0.6515, 0.5683]`), and every row of the weight matrix sums to 1.

### 2. `self-attention.ipynb` — Trainable Self-Attention

The input is projected with three weight matrices (`W_query`, `W_key`, `W_value`) with `d_in = 3` and `d_out = 2`.

- First, the computation is done for one token step by step: query, keys, values, scores, scaled softmax, and context vector.
- Then it is packed into `SelfAttentionV1`, which uses `nn.Parameter` matrices.
- Then it is rewritten as `SelfAttentionV2`, which uses `nn.Linear` layers with `qkv_bias=False`.
- Exercise 1 copies the weights of `SelfAttentionV2` into `SelfAttentionV1`. The stored output of `SelfAttentionV1` then matches the output of `SelfAttentionV2`.

Why it matters: this is the layer that real models train. `nn.Linear` is preferred because it has a better weight initialization than plain `torch.rand`, and the two versions can be made to give the same result, as Exercise 1 shows.

### 3. `casual-attention.ipynb` — Causal Attention

A language model must not look at future tokens. This notebook shows how to enforce that:

- a lower-triangular mask (`torch.tril`) multiplied with the weights, followed by re-normalization of each row;
- a more efficient method: fill the upper-triangular scores with `-inf` (`masked_fill`) before softmax;
- `nn.Dropout(0.5)`, first on a matrix of ones, then on the attention weights;
- a batch built with `torch.stack` (shape `[2, 6, 3]`);
- the `CasualAttention` class with a registered `mask` buffer, which accepts batched input.

The stored output of the `-inf` method gives the same weights as the multiply-and-renormalize method (for example, the first row is `[1, 0, 0, 0, 0, 0]`). The class run ends with context vectors of shape `[2, 6, 2]`.

### 4. `multi-head-attention.ipynb` — Multi-Head Attention

Two ways to build multi-head attention are compared:

- **`MultiHeadAttentionWrapper`** stacks several `CasualAttention` modules and joins their outputs with `torch.cat`. With `d_out = 2` and 2 heads, the output shape is `[2, 6, 4]`. Exercise 3.2 changes `d_out` to 1 and gets shape `[2, 6, 2]`.
- **`MultiHeadAttention`** uses one set of projections and splits the result into heads with `view` and `transpose`. It adds an output projection (`out_proj`) and checks that `d_out` is divisible by `num_heads`.

A small example with a 4-dimensional tensor shows that one batched matrix product `a @ a.transpose(2, 3)` gives the same numbers as computing each head by itself.

Exercise 3.3 creates a `MultiHeadAttention` with GPT-2 small settings (`d_in = d_out = 768`, `context_length = 1024`, `num_heads = 12`). The cell has no stored output.

---

## Key Concepts

| Concept | Meaning in these notebooks |
|---------|----------------------------|
| Attention score | Dot product between a query and a key (or between two input vectors in notebook 1) |
| Attention weight | Score after softmax; each row sums to 1 |
| Context vector | Weighted sum of the value vectors (or input vectors in notebook 1) |
| Scaling | Scores are divided by `d_k ** 0.5` before softmax |
| Causal mask | Upper-triangular mask that blocks future tokens |
| Dropout | Randomly sets attention weights to zero; the remaining weights are scaled up (for `p = 0.5`, by 2) |
| Head | One independent attention computation; `head_dim = d_out // num_heads` |

---

## Concept Flow

```text
inputs (tokens x d_in)
   │
   ├─ notebook 1: scores = inputs @ inputs.T ──► softmax ──► weights @ inputs
   │
   ├─ notebook 2: Q, K, V = x @ W_q, W_k, W_v
   │              scores = Q @ K.T / sqrt(d_k) ──► softmax ──► weights @ V
   │
   ├─ notebook 3: scores ──► causal mask (-inf) ──► softmax ──► dropout ──► weights @ V
   │
   └─ notebook 4: split Q, K, V into heads ──► causal attention per head
                  ──► join heads ──► out_proj
```

---

## Folder Structure

```text
chap03/
├── README.md
├── attention-mechanism.ipynb
├── self-attention.ipynb
├── casual-attention.ipynb
└── multi-head-attention.ipynb
```

Jupyter may also create a hidden `.ipynb_checkpoints/` folder here. It contains automatic copies of the notebooks and is not part of the lessons.

---

## Environment and Setup

The notebooks use Python and PyTorch only (`torch` and `torch.nn`). They run in a `Python 3 (ipykernel)` kernel, so you need Jupyter.

This folder has no `requirements.txt` or `environment.yml`, and no package versions are recorded. A simple setup is:

```bash
pip install torch jupyterlab
```

The notebooks do not download data or models, and they do not need internet access, credentials, or a GPU. All input data is typed directly in the first cell.

## How to Run

```bash
cd chap03
jupyter lab
```

Open the notebooks in the order shown above and run the cells from top to bottom. Each notebook is independent: it defines the classes it needs, so you can run them in any order, but the study order is easier to follow.

---

## Stored Outputs

Each notebook keeps the outputs of its last run (printed tensors and shapes only; there are no plots). They help you check your own results. Values that depend on random weights can change if you change a seed or the PyTorch version, so treat them as examples, not as fixed targets.

---

## Important Notes

These notes come from reading the saved code and outputs. The notebooks were not re-run to write this README.

- **Variable names in `self-attention.ipynb`:** the cells that print the results of the two classes call `sa1(inputs)` and `sa2(inputs)`, but the objects are created as `sa_v1` and `sa_v2`. The stored outputs are from an earlier run. If you run the notebook from top to bottom, those two cells raise a `NameError` until the names match.
- **Mask in the classes:** in `CasualAttention.forward` and `MultiHeadAttention.forward`, the line `attn_scores.masked_fill(...)` is not assigned to a variable, and `masked_fill` does not change the tensor in place. As written, the mask is not applied inside these two classes. The masking in the earlier cells of `casual-attention.ipynb` is applied correctly. To apply the mask in a class, assign the result back (or use the in-place `masked_fill_`).
- **Stored multi-head outputs:** because of the point above, the numbers printed for the multi-head classes come from attention without a working mask.
- **Exercise 3.3** (GPT-2 sized layer) only creates the layer. It does not print a result.
- **No Markdown cells:** the notebooks contain only code cells and comments. This README is the written explanation of the chapter.

---

## Reference

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning Publications, 2024. Chapter 3, "Coding Attention Mechanisms". See the [publisher listing at O'Reilly](https://oreilly.com/library/view/build-a-large/9781633437166).

## Disclaimer

These notebooks are my own study work based on the book above. They are not official material from the author or the publisher, and they do not copy the book's text.

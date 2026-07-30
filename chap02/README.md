# Chapter 2 — Working with Text Data

This chapter builds the text-processing pipeline required before a GPT-style language model can learn from text. The notebooks progress from a transparent word-and-punctuation tokenizer to GPT-2 byte pair encoding, sliding input-target windows, PyTorch data loading, and the token-plus-position embeddings consumed by later transformer layers.

The annotated notebooks preserve the original code, stored outputs, execution counts, and notebook metadata. Explanatory Markdown, focused comments, and Google-style docstrings were added without intentionally changing computational behavior.

> This is a personal educational implementation based on Chapter 2 of Sebastian Raschka's *Build a Large Language Model (From Scratch)*. It is not the official book repository. The official companion code is maintained at https://github.com/rasbt/LLMs-from-scratch.

## Chapter goal

The goal is to understand every representation between raw text and model-ready vectors:

```text
raw text
→ tokens
→ vocabulary IDs
→ BPE token IDs
→ sliding input-target windows
→ batched tensors
→ token embeddings + positional embeddings
```

Each notebook isolates one part of this path so that the intermediate values, shapes, assumptions, and failure modes remain visible.

## Learning objectives

After completing this chapter, you should be able to:

- split text into words and punctuation with regular expressions;
- build forward and reverse token-ID vocabularies;
- explain unknown-token and end-of-text handling;
- encode and decode text with the GPT-2 tokenizer;
- describe how BPE represents unfamiliar text with subword pieces;
- form one-token-shifted targets for causal language modeling;
- control context length, stride, overlap, batching, and shuffling;
- implement a PyTorch `Dataset` and `DataLoader` for token windows;
- map token IDs to trainable token and positional embeddings;
- track batch, sequence, and embedding dimensions through the pipeline.

## Recommended study order

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | [`simple_tokenizer.ipynb`](simple_tokenizer.ipynb) | Build a small tokenizer from scratch, observe the out-of-vocabulary failure in Version 1, and add `<\|unk\|>` and `<\|endoftext\|>` handling in Version 2. |
| 2 | [`bpe_tokenizer.ipynb`](bpe_tokenizer.ipynb) | Use GPT-2 tokenization with `tiktoken`, create shifted context-target pairs, and package sliding token windows with PyTorch. |
| 3 | [`bpe-from-scratch-simple.ipynb`](bpe-from-scratch-simple.ipynb) | Optional deeper study of how BPE learns and applies merge rules. This notebook already contained a detailed educational narrative and was intentionally left unchanged. |
| 4 | [`embedding.ipynb`](embedding.ipynb) | Convert batched token IDs into token embeddings, create positional embeddings, and combine both representations. |

The annotated notebooks now use the canonical chapter filenames. Their pre-annotation versions remain recoverable from Git history.

## Key concepts

### Tokenization and vocabulary construction

`simple_tokenizer.ipynb` makes the vocabulary interface explicit. Regular expressions split the source text into words and punctuation, unique tokens receive integer IDs, and reverse lookup reconstructs readable text.

The two tokenizer versions demonstrate a real design choice:

| Tokenizer | Unseen-token behavior | Educational value |
|---|---|---|
| `SimpleTokenizerV1` | Raises `KeyError` | Exposes the limitation of a closed vocabulary |
| `SimpleTokenizerV2` | Replaces unseen tokens with `<\|unk\|>` | Demonstrates explicit fallback and document boundaries |
| GPT-2 `tiktoken` | Uses learned byte-pair pieces | Represents arbitrary text without relying on one unknown-word marker |

### Next-token prediction data

For a token sequence \(t_0, t_1, \ldots\), the training target is the input shifted forward by one position:

$$
\mathbf{x} = (t_i, \ldots, t_{i+L-1}),
\qquad
\mathbf{y} = (t_{i+1}, \ldots, t_{i+L})
$$

`GPTDatasetV1` creates these pairs with a configurable context length and stride. `DataLoader` then groups examples into batches suitable for a later training loop.

### Token and positional embeddings

Token IDs are categorical indices, so they are converted into dense trainable vectors. A separate positional table represents sequence locations. The model input is their element-wise sum:

$$
\mathbf{E}_{\text{input}} =
\mathbf{E}_{\text{token}} + \mathbf{E}_{\text{position}}
$$

The preserved embedding notebook output traces the shapes from `[8, 4]` token IDs to `[8, 4, 256]` combined vectors.

## Files

```text
chap02/
├── README.md
├── text-verdict.txt
├── simple_tokenizer.ipynb
├── bpe_tokenizer.ipynb
├── bpe-from-scratch-simple.ipynb
└── embedding.ipynb
```

| File | Role |
|---|---|
| `text-verdict.txt` | Local chapter corpus downloaded from the official companion repository. |
| `simple_tokenizer.ipynb` | Annotated simple-tokenizer lesson with the original outputs preserved. |
| `bpe_tokenizer.ipynb` | Annotated GPT-2 tokenizer and data-loading lesson; also imported by `embedding.ipynb`. |
| `embedding.ipynb` | Annotated token and positional embedding lesson with the original outputs preserved. |
| `bpe-from-scratch-simple.ipynb` | Existing standalone BPE tutorial; excluded from the new annotation pass. |

## Environment

This chapter does not currently include a chapter-specific dependency manifest. The notebooks directly use:

- PyTorch;
- `tiktoken`;
- `import-ipynb`;
- JupyterLab;
- Python standard-library modules such as `re` and `urllib.request`.

Create and activate a virtual environment, then install the verified core packages:

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Git Bash / Linux / macOS
source .venv/bin/activate
```

```bash
pip install torch tiktoken import-ipynb jupyterlab
```

No package versions are pinned in this chapter, so exact output formatting may vary across environments.

## Running the notebooks

From the repository root:

```bash
cd chap02
jupyter lab
```

Open the annotated notebooks in the recommended order and run them from top to bottom when you want to reproduce the stored results.

Important execution notes:

- `simple_tokenizer.ipynb` downloads `the-verdict.txt` on its first cell and writes it as `text-verdict.txt`. Internet access is required for that download.
- `bpe_tokenizer.ipynb` reads the local `text-verdict.txt` file.
- `embedding.ipynb` uses `import_ipynb` to import `create_dataloader` and `raw_text` from `bpe_tokenizer.ipynb`. Keep both files in the same folder.
- Importing a notebook through `import-ipynb` executes its cells as a module. The embedding notebook uses `%%capture` to suppress that import-time output.
- The embedding data loader uses shuffling, so rerunning it can produce different token batches even though the recorded tensor shapes remain the same.

## Preserved outputs and known teaching checkpoints

The annotated notebooks were not re-executed. Their existing outputs remain part of the learning material:

| Notebook | Preserved evidence |
|---|---|
| `simple_tokenizer.ipynb` | Corpus preview, token counts, vocabulary entries, encoded IDs, decoded text, and the intentional `KeyError: 'Hello'`. |
| `bpe_tokenizer.ipynb` | GPT-2 token IDs, decoded subword pieces, corpus token count, context-target examples, and batched tensors. |
| `embedding.ipynb` | Seeded embedding weights, token batches, and token, position, and combined embedding shapes. |
| `bpe-from-scratch-simple.ipynb` | BPE training, merge, encoding, decoding, and round-trip demonstrations from the existing tutorial. |

The error in `SimpleTokenizerV1` is intentionally preserved because it demonstrates why unknown-token handling is needed. It is addressed later in the same notebook by `SimpleTokenizerV2`; the stored traceback was not deleted or replaced.

## Suggested experiments

After completing the notebooks once:

1. Extend the regular-expression tokenizer with another punctuation rule and inspect the vocabulary change.
2. Encode several words absent from `text-verdict.txt` with both simple tokenizer versions.
3. Compare GPT-2 token pieces for common and invented words.
4. Change `max_length` and `stride` to compare overlapping and non-overlapping windows.
5. Disable data-loader shuffling while tracing specific token windows.
6. Change the embedding dimension and predict every resulting tensor shape before running the cells.

## Key takeaway

Language models do not consume raw text. They learn from numerical sequences whose construction determines vocabulary coverage, context boundaries, target alignment, and tensor shape. This chapter makes that full preparation pipeline inspectable before attention or model training is introduced.

## References

- [Official *LLMs from Scratch* repository](https://github.com/rasbt/LLMs-from-scratch)
- [Official Chapter 2 companion code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/01_main-chapter-code)
- [OpenAI `tiktoken`](https://github.com/openai/tiktoken)
- [PyTorch data loading utilities](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Python regular-expression documentation](https://docs.python.org/3/library/re.html)
- [Book page — *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch)

# PyTorch Appendix — Foundations for LLM Implementation

This section contains the annotated PyTorch appendix notebook for the *Build a Large Language Model from Scratch* study repository. It focuses on the core PyTorch skills needed before implementing tokenizers, attention, transformer blocks, pretraining loops, and fine-tuning workflows.

The notebook has been turned into a teaching notebook: it includes an opening overview, markdown explanations before the code cells, and comments inside the code. The original visible outputs are preserved because they are part of the learning value of the notebook.

## Why this section matters

Before building an LLM, it is important to be comfortable with the framework mechanics that make the model trainable. A GPT-style language model is built from tensor operations, matrix multiplications, neural network modules, gradients, dataloaders, optimizer steps, checkpoints, and device-aware training. This appendix turns those ideas into small examples that are easy to inspect and debug.

## Files

```text
pytorch_appendix/
├── README.md
├── pytorch-appendix.ipynb
├── multiple_gpus.py
└── pytorch_appendix.pth          # generated after running the notebook
```

| File | Purpose |
|---|---|
| `pytorch-appendix.ipynb` | Main annotated notebook using the original filename. Outputs are preserved. |
| `multiple_gpus.py` | Script version of the Distributed Data Parallel example, intended to be run from the terminal instead of inside a notebook. |
| `pytorch_appendix.pth` | Model checkpoint produced when the notebook saves the toy model weights. This file is generated, so it does not need to be committed unless desired. |

## Topics covered

### 1. Tensor fundamentals

The notebook starts with scalar, vector, matrix, and 3D tensors. It demonstrates dtype inference, dtype conversion, shape inspection, reshaping, transposition, and matrix multiplication.

These operations are the foundation of later LLM components. Token batches, embeddings, attention scores, transformer activations, and logits are all tensors with carefully managed shapes.

### 2. Automatic differentiation

A tiny one-neuron example shows how PyTorch tracks gradients through a computation graph. The notebook compares explicit gradient extraction with `torch.autograd.grad` and the standard training-loop approach using `loss.backward()`.

### 3. Neural network modules

The notebook defines a small `torch.nn.Module` with linear layers and ReLU activations. It shows how to instantiate a model, inspect its structure, count trainable parameters, run inference, and convert logits into probabilities.

### 4. Dataset and DataLoader workflow

A small toy classification dataset is wrapped in a custom `Dataset`, then batched with a `DataLoader`. This prepares the same pattern that later text datasets will use after tokenization.

### 5. Training loop

The notebook walks through the standard PyTorch training loop:

```text
forward pass → loss → zero gradients → backward pass → optimizer step
```

It also includes evaluation mode, `torch.no_grad()`, class prediction, and a reusable accuracy function.

### 6. Saving and loading weights

The model is saved with `state_dict`, which is the recommended PyTorch pattern for storing learned parameters. The notebook then recreates the architecture and loads the saved weights.

### 7. CPU/GPU device handling

The CUDA section demonstrates how to check GPU availability, move tensors to the GPU, and understand CPU/GPU mismatch errors. One output intentionally shows a device mismatch error, which is useful because it teaches why tensors and models must live on the same device.

### 8. Distributed Data Parallel overview

The notebook includes DDP setup code for learning purposes, while `multiple_gpus.py` provides a cleaner script-style version. DDP is usually more reliable from a terminal script than from an interactive notebook.

## Running the notebook

From the repository root:

```bash
cd pytorch_appendix
jupyter lab pytorch-appendix.ipynb
```

Run the notebook from top to bottom when you want to reproduce the results. The committed notebook already contains the original outputs, so readers can inspect the results even before running the code locally.

A few GPU cells depend on CUDA. If you run the notebook on a CPU-only machine, those outputs will differ unless you adapt the CUDA-specific cells.

## Running the multi-GPU script

DDP is designed to be launched from the terminal. On a machine with multiple CUDA GPUs, run:

```bash
python multiple_gpus.py
```

For serious multi-GPU experiments, prefer Linux or WSL with a CUDA-enabled PyTorch build. Notebook-based DDP can be unreliable because multiprocessing behaves differently across operating systems and interactive kernels.

## Expected learning outcome

After completing this appendix, you should be comfortable with:

- Creating and reshaping PyTorch tensors.
- Understanding dtype and device placement.
- Computing gradients with autograd.
- Defining simple neural networks with `nn.Module`.
- Creating custom datasets and dataloaders.
- Writing a complete training loop.
- Saving and loading model weights.
- Moving models and tensors between CPU and GPU.
- Understanding the high-level structure of DDP training.

## Connection to later LLM chapters

The later chapters scale these same ideas up:

| PyTorch appendix concept | Later LLM use |
|---|---|
| Tensor shapes | Token batches, embeddings, attention scores, logits |
| Matrix multiplication | Linear projections, attention, feed-forward layers |
| Autograd | Pretraining and fine-tuning loss optimization |
| `nn.Module` | Attention modules, transformer blocks, GPT model |
| DataLoader | Tokenized text batches for next-token prediction |
| Device handling | GPU training and inference |
| `state_dict` | Checkpoints and pretrained weight loading |

This appendix is therefore the foundation layer of the full repository.

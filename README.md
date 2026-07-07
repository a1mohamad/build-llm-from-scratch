# Build a Large Language Model from Scratch — Study Repository

An educational, chapter-by-chapter implementation workspace based on Sebastian Raschka's *Build a Large Language Model (From Scratch)*. This repository is organized as a practical learning lab: notebooks explain the concepts, Python scripts turn important pieces into reusable code, and each section documents what was implemented and why it matters.

> This is a personal educational repository. It is not the official book repository. The official companion code is maintained by Sebastian Raschka at: https://github.com/rasbt/LLMs-from-scratch

## Project goal

The purpose of this repository is to understand large language models from the inside out by implementing the main components directly in PyTorch. Instead of treating LLMs as black-box APIs, the project builds the foundations step by step: tensors, tokenization, embeddings, attention, transformer blocks, pretraining, evaluation, and fine-tuning.

By the end of the full project, the repository should make it clear how a GPT-style model processes text, learns from next-token prediction, and can be adapted for downstream tasks.

## Learning path

The repository follows the natural progression of the book:

1. **PyTorch foundations** — tensors, autograd, neural network modules, dataloaders, training loops, device placement, and multi-GPU basics.
2. **Text data preparation** — tokenization, token IDs, sliding windows, input-target pairs, and batching for language modeling.
3. **Attention mechanisms** — self-attention, causal attention, multi-head attention, and masking.
4. **GPT-style architecture** — embeddings, transformer blocks, layer normalization, feed-forward networks, residual connections, and output heads.
5. **Pretraining** — next-token prediction, loss tracking, generation, checkpointing, and training loops.
6. **Model loading and reuse** — working with pretrained weights and comparing custom implementations with known model configurations.
7. **Fine-tuning** — adapting the pretrained model for classification and instruction-following tasks.
8. **Appendices and engineering notes** — PyTorch, training utilities, performance tips, and implementation details.

## Repository structure

```text
build-LLM-from-scratch/
├── README.md
├── pytorch_appendix/
│   ├── README.md
│   ├── pytorch-appendix.ipynb
│   ├── pytorch-appendix-annotated.ipynb
│   ├── multiple_gpus.py
│   └── pytorch_appendix.pth          # generated after running the notebook
├── chapter_02_tokenization/          # planned / future section
├── chapter_03_attention/             # planned / future section
├── chapter_04_gpt_model/             # planned / future section
├── chapter_05_pretraining/           # planned / future section
├── chapter_06_classification/        # planned / future section
└── chapter_07_instruction_finetuning/ # planned / future section
```

The exact folder names can evolve as the project grows, but each section should keep the same documentation style: a local README, annotated notebooks, reusable scripts, and generated artifacts excluded from version control when appropriate.

## Current status

| Section | Status | Description |
|---|---:|---|
| PyTorch Appendix | Done / annotated | Foundation notebook with tensor operations, autograd, modules, dataloaders, training loop, saving/loading, CUDA, and DDP notes. |
| Tokenization and data loading | Planned | Text preprocessing and input-target generation for language modeling. |
| Attention | Planned | Self-attention, causal masking, and multi-head attention. |
| GPT model | Planned | Transformer block and full GPT-style architecture. |
| Pretraining | Planned | Training the language model with next-token prediction. |
| Fine-tuning | Planned | Classification and instruction-following adaptation. |

## How to use this repository

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/build-LLM-from-scratch.git
cd build-LLM-from-scratch
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Git Bash / Linux / macOS
source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install torch jupyterlab notebook
```

Then start JupyterLab:

```bash
jupyter lab
```

Open the section notebook you want to study, run it from top to bottom, and read the markdown explanations before each code cell.

## Documentation style

Each notebook in this repository should follow this pattern:

- A large opening markdown cell explaining the goal of the notebook.
- Markdown before each important code cell.
- Clear comments inside code cells, especially around tensor shapes, training loops, and device movement.
- Small examples first, then reusable functions or scripts.
- Practical notes for CPU, GPU, and Windows/Notebook limitations.

This style makes the repository useful both as a personal learning record and as a portfolio-quality educational project.

## Notes on generated files

Some files are created when notebooks are executed, such as model checkpoints (`*.pth`) or cache folders. These should usually be excluded from Git unless they are intentionally small and useful for demonstration.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
.ipynb_checkpoints/
.venv/
*.pth
*.pt
*.ckpt
```

## References

- Official book code repository: https://github.com/rasbt/LLMs-from-scratch
- Book page: https://www.manning.com/books/build-a-large-language-model-from-scratch
- Author resources: https://sebastianraschka.com/llms-from-scratch/

## License

Add the license that matches your repository goals. If this is only your own educational code and notes, the MIT License is a common choice. Do not copy book text or official repository files unless their license permits it.

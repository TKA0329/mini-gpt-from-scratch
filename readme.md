 MiniGPT — Transformer Language Model from Scratch

A small GPT-style language model implemented from scratch in PyTorch, trained on ~1M lines of text using a custom SentencePiece tokenizer.

## Overview
This project is a from-scratch implementation of a GPT-style autoregressive Transformer.
I built it to understand the full training pipeline of language models, including
tokenization, causal self-attention, training dynamics, and sampling strategies.

The model is intentionally small and trained with limited compute, focusing on clarity and correctness rather than performance.

## Key Features
- Custom **SentencePiece BPE tokenizer** (20k vocab)
- GPT-style **causal Transformer** with:
  - Multi-head self-attention
  - Residual connections
  - LayerNorm + GELU
- **Weight tying** between embedding and output projection
- **Top-k sampling** with temperature control
- Learning rate **warmup + cosine decay**
- Gradient clipping and NaN checks
- Fully written in **PyTorch (no HuggingFace abstractions)**

## Model Architecture
- Layers: 8 Transformer blockss
- Embedding dimension: 384
- Attention heads: 8
- Context length: 512 tokens
- Vocabulary size: 20,000
- Dropout: 0.1

## Dataset
The model was trained on approximately **1 million lines of text** compiled from
public-domain sources (Project Gutenberg), with basic cleaning applied.

The goal was not perfect text quality, but to study how a Transformer learns structure
from imperfect real-world data.

## Training
- Optimizer: AdamW
- Learning rate: 3e-4
- Scheduler: Linear warmup (10%) + cosine decay
- Batch size: 16
- Epochs: 5
- Loss: Cross-entropy (next-token prediction)

Training was done on limited compute, so the model is not fully converged.

## Limitations
- Small model size and limited training data
- No reinforcement learning or instruction tuning
- No evaluation on downstream benchmarks

This project is intended as a learning and research exercise rather than a production model.

## Motivation

This project was built to deepen my understanding of language models by implementing every major component manually, from tokenization to sampling.
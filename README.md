# Belief State Objective Function

## Objective

This repository provides an implementation of a **belief state objective function** that leverages forward and backward state representations to compute loss functions for sequence-based models. The function utilizes PyTorch to perform gradient calculations over all prefix/suffix pairs, making it useful for natural language processing (NLP) tasks involving autoregressive transformers.

## Prerequisites

Before running the code, ensure that the following dependencies are installed:

1. **PyTorch**: The framework used for deep learning computations.
2. **NumPy**: Used for numerical operations.

## Installation Steps

### Step 1: Install PyTorch

Follow the official PyTorch installation guide based on your system configuration: [PyTorch Installation](https://pytorch.org/get-started/locally/)

```bash
pip install torch
```

### Step 2: Install Additional Dependencies
To install the required libraries, run:

```bash
pip install numpy
```

### Usage
The belief state objective function is implemented in main_objective.py. You can integrate this function into your own NLP models to compute loss over sequence pairs.

Run the script to test the implementation:

```bash
python main_objective.py
```

### How It Works
The function belief_state_objective takes as input:

1. Forward state (all_f) and backward state (all_b).
    * A text prediction head (text_head).
    * The input sequence (x).
    * It generates time step combinations and filters pairs that meet the required constraints.

2. Using these indices, it extracts forward and backward representations.

3. The function computes the loss using CrossEntropyLoss, optimizing over prefix/suffix pairs.

4. The implementation supports backpropagation and gradient updates.

### Model Integration
This repository does not provide a full NLP training pipeline. Instead, it offers a reusable objective function that can be incorporated into your existing sequence-based models.

### References
If you use this repository in your work, please cite the corresponding paper:

> **The Belief State Transformer** - Edward S. Hu and Kwangjun Ahn and Qinghua Liu and Haoran Xu and Manan Tomar and Ada Langford and Dinesh Jayaraman and Alex Lamb and John Langford, 2025. [[Link to Paper]](https://arxiv.org/abs/2410.23506)

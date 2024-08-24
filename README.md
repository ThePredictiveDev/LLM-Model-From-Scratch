# Basic Transformer-Based LLM Coded From Scratch

## Overview

This repository contains a foundational implementation of a Large Language Model (LLM) built from scratch using the Transformer algorithm. The code is designed to be educational and provide a clear understanding of the core components of modern language models.

## Features

- **Transformer Architecture**: Basic implementation of the Transformer model.
- **Training Pipeline**: Includes data reading, batch preparation, and training with loss estimation.
- **Text Generation**: Functionality to generate text based on the trained model.
- **Evaluation**: Basic loss evaluation on training and validation datasets.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python libraries: `torch`

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/ThePredictiveDev/LLM-Model-From-Scratch.git
    cd LLM-Model-From-Scratch
    ```

2. **Install Dependencies**

    ```bash
    pip install torch
    ```

### Usage

1. **Data Preparation**

    Update the path to your dataset in the `Reading Data` section of the code:

    ```python
    with open(r'C:\Users\Devansh\Downloads\LLM FROM SCRATCH\train.json', 'r', encoding='utf-8') as f:
        text = f.read()
    ```


2. **Generating Text**

    After training, you can generate text using the trained model. Example code snippet for text generation:

    ```python
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ x

    print(out.shape)
    ```

## License

 This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

1. Inspired by the original Transformer paper by Vaswani et al.
2. Built using PyTorch for ease of implementation and flexibility.




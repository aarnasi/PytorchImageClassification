# PyTorch FashionMNIST Classification Example

This project demonstrates a basic end-to-end workflow for training and evaluating a simple feed-forward neural network for image classification using PyTorch and the FashionMNIST dataset. 

## Features

* **Dataset Handling:** Automatically downloads and prepares the FashionMNIST dataset using `torchvision.datasets`.
* **Data Loading:** Uses `torch.utils.data.DataLoader` for efficient batching and shuffling.
* **Model Definition:** Defines a simple Neural Network with Linear layers and ReLU activations using `torch.nn.Module`.
* **Device Agnostic:** Automatically detects and utilizes available hardware accelerators (like CUDA GPUs or Apple Silicon MPS) or defaults to CPU.
* **Training:** Implements a standard training loop with loss calculation (CrossEntropyLoss), backpropagation, and optimization (SGD).
* **Evaluation:** Implements an evaluation loop to measure accuracy and loss on the test set.
* **Logging:** Uses Python's `logging` module for informative output during execution, replacing standard `print` statements.
* **Model Persistence:** Saves the trained model's state dictionary (`model.pth`).
* **Model Loading:** Demonstrates loading the saved state dictionary into a new model instance.
* **Inference:** Shows how to perform inference on a single sample from the test set using the loaded model.

## Requirements

* Python (3.8+ recommended)
* PyTorch
* Torchvision

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone https://github.com/aarnasi/PytorchImageClassification.git # If applicable
    cd PytorchImageClassification
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install PyTorch and Torchvision:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your specific system (OS, package manager, CUDA version if applicable). A common command is:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure this command installs the version compatible with your hardware if you intend to use GPU acceleration.)*

## Usage

Run the script from your terminal:

```bash
python trainer.py
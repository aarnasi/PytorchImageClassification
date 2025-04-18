"""
PyTorch FashionMNIST Classification Example

This script demonstrates a basic PyTorch workflow for image classification
using the FashionMNIST dataset. It covers the following steps:

1.  Data Loading: Downloads and prepares the FashionMNIST training and test datasets.
2.  Data Loaders: Creates DataLoader instances for batching and shuffling.
3.  Model Definition: Defines a simple feed-forward neural network using nn.Module.
4.  Device Setup: Detects and selects the appropriate device (CPU or GPU/Accelerator).
5.  Training Setup: Initializes the loss function (CrossEntropyLoss) and optimizer (SGD).
6.  Training Loop: Implements the training process over multiple epochs, including
    forward pass, loss calculation, backpropagation, and optimizer step.
7.  Testing Loop: Implements the evaluation process on the test dataset using
    torch.no_grad() for efficiency.
8.  Model Saving: Saves the trained model's state dictionary to a file.
9.  Model Loading: Loads the saved model state dictionary into a new model instance.
10. Inference: Performs prediction on a single sample from the test set using the loaded model.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import logging # Import the logging library

# --- Logging Setup ---
# Configure basic logging:
# - level=logging.INFO: Log messages with level INFO and above (WARNING, ERROR, CRITICAL).
# - format: Define the structure of the log messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Loading ---

logging.info("Starting data loading...")
# Download and load the training data from the FashionMNIST dataset.
training_data = datasets.FashionMNIST(
    root="data",        # Directory where the data will be stored/downloaded
    train=True,         # Specify that this is the training set
    download=True,      # Download the dataset if it's not already present
    transform=ToTensor() # Convert images to PyTorch Tensors (scales pixel values to [0, 1])
)

# Download and load the test data from the FashionMNIST dataset.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,        # Specify that this is the test set
    download=True,
    transform=ToTensor()
)
logging.info("Data loading complete.")

# --- Data Loaders ---

batch_size = 64  # Define the number of samples per batch

# Create data loaders. These wrap the datasets and provide an iterable
# over batches of data, handling shuffling (default for train=True) and batching.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
logging.info(f"DataLoaders created with batch size: {batch_size}")

# --- Data Inspection (Optional) ---

# Iterate once through the test dataloader to check the shape of features (X) and labels (y).
# This helps understand the data structure.
logging.info("Inspecting first batch of test data...")
for X, y in test_dataloader:
    logging.info(f"Shape of X [N, C, H, W]: {X.shape}")  # N: Batch size, C: Channels, H: Height, W: Width
    logging.info(f"Shape of y: {y.shape} {y.dtype}")     # Shape and data type of the labels
    break  # Exit after inspecting the first batch

# --- Device Setup ---

# Get the appropriate device for training (GPU if available via CUDA/MPS, otherwise CPU).
# Uses the newer torch.accelerator API for broader hardware support (like MPS on Apple Silicon).
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
logging.info(f"Using {device} device")

# --- Model Definition ---

# Define the neural network model by subclassing nn.Module.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # Flattens the 2D image (C, H, W) into a 1D vector (C*H*W)
        # Define a sequential container of layers.
        # Data flows sequentially through these layers.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # Input layer (784 features from flattened 28x28 image) to 512 hidden units
            nn.ReLU(),              # Rectified Linear Unit activation function
            nn.Linear(512, 512),    # Hidden layer (512 units) to another hidden layer (512 units)
            nn.ReLU(),
            nn.Linear(512, 10)      # Output layer (512 units) to 10 output units (one for each class)
        )

    # Define the forward pass of the network.
    # This specifies how input data `x` flows through the defined layers.
    def forward(self, x):
        x = self.flatten(x)          # Flatten the input tensor
        logits = self.linear_relu_stack(x)  # Pass the flattened tensor through the linear stack
        return logits                # Return the raw output scores (logits)

# --- Model Instantiation ---

# Create an instance of the NeuralNetwork.
model = NeuralNetwork()
# Move the model's parameters and buffers to the selected device (GPU or CPU).
model = model.to(device)
logging.info(f"Model Architecture:\n{model}")  # Log the model architecture (uses __str__)

# --- Loss Function and Optimizer ---

# Define the loss function: CrossEntropyLoss is common for multi-class classification.
loss_fn = nn.CrossEntropyLoss()
logging.info(f"Loss function: {type(loss_fn).__name__}")

# Define the optimizer: Stochastic Gradient Descent (SGD).
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
logging.info(f"Optimizer: {type(optimizer).__name__}, Learning Rate: {optimizer.defaults['lr']}")


# --- Training Function ---

# Define the training loop function for one epoch.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # Total number of samples in the training dataset
    model.train()  # Set the model to training mode
    # Iterate over batches of data provided by the dataloader.
    for batch, (X, y) in enumerate(dataloader):
        # Move the input features (X) and labels (y) to the configured device.
        X, y = X.to(device), y.to(device)

        # --- Forward Pass ---
        pred = model(X)
        loss = loss_fn(pred, y)

        # --- Backpropagation ---
        loss.backward()

        # --- Optimizer Step ---
        optimizer.step()

        # --- Zero Gradients ---
        optimizer.zero_grad()

        # Log progress every 100 batches.
        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# --- Testing/Evaluation Function ---

# Define the testing/evaluation loop function.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Total number of samples in the test dataset
    num_batches = len(dataloader)   # Total number of batches in the test dataloader
    model.eval()   # Set the model to evaluation mode
    test_loss, correct = 0, 0  # Initialize total loss and number of correct predictions
    # Disable gradient calculations during evaluation
    with torch.no_grad():
        # Iterate over batches of data.
        for X, y in dataloader:
            # Move data to the configured device.
            X, y = X.to(device), y.to(device)
            # Perform a forward pass.
            pred = model(X)
            # Accumulate the loss for this batch.
            test_loss += loss_fn(pred, y).item()
            # Count the number of correct predictions in this batch.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Calculate average loss and accuracy over all batches.
    test_loss /= num_batches
    correct /= size
    # Log the test results
    logging.info(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

# --- Training Loop ---

epochs = 5  # Number of times to iterate over the entire training dataset
logging.info(f"Starting training for {epochs} epochs...")
# Loop over the specified number of epochs.
for t in range(epochs):
    logging.info(f"Epoch {t + 1}\n-------------------------------")
    # Call the training function for the current epoch.
    train(train_dataloader, model, loss_fn, optimizer)
    # Call the testing function to evaluate the model.
    test(test_dataloader, model, loss_fn)
logging.info("Done training!")

# --- Model Saving ---

# Save the trained model's state dictionary.
save_path = "model.pth"
torch.save(model.state_dict(), save_path)
logging.info(f"Saved PyTorch Model State to {save_path}")

# --- Model Loading ---

# Create a new instance of the model architecture.
model_loaded = NeuralNetwork().to(device) # Move the new model instance to the device as well

# Load the saved state dictionary into the new model instance.
model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
logging.info(f"Loaded model state from {save_path}")

# --- Inference/Prediction Example ---

logging.info("Performing inference on a sample...")
# Define the human-readable names for the FashionMNIST classes.
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Set the loaded model to evaluation mode.
model_loaded.eval()
# Get a single sample from the test dataset.
x, y = test_data[0][0], test_data[0][1]

# Perform inference within torch.no_grad() context.
with torch.no_grad():
    x = x.to(device)
    pred = model_loaded(x)
    predicted_idx = pred[0].argmax(0)
    predicted, actual = classes[predicted_idx], classes[y]
    logging.info(f'Predicted: "{predicted}", Actual: "{actual}"')
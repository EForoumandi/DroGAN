"""
Configuration for DroGAN.

This configuration file sets up the environment for training the DroGAN model, 
including directories, hyperparameters, device setup, and model saving/loading options.

"""

import torch

# Define device: Use GPU if available, else fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory configurations
Input_DIR = # Directory to input data files
Target_DIR = # Directory to input data files

# Training hyperparameters
LEARNING_RATE = # learning rate
BATCH_SIZE = # batch size
NUM_WORKERS = # number of workers
IMAGE_SIZE = # Input image size
L1_LAMBDA = # Weight for L1 loss 

# Model training configuration
NUM_EPOCHS = # number of epochs
LOAD_MODEL = # Flag to load the model
SAVE_MODEL = # Flag to save the model

# Checkpoint paths
CHECKPOINT_DISC = "disc.pth.tar"  # Discriminator checkpoint file
CHECKPOINT_GEN = "gen.pth.tar"    # Generator checkpoint file

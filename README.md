
# Variational Autoencoder (VAE) on MNIST

This project implements a **Variational Autoencoder (VAE)** using TensorFlow and Keras to learn a 2D latent representation of the MNIST handwritten digits dataset. The model is trained to reconstruct input images while learning a meaningful latent space for generative sampling.

---

## 🔍 Overview

- **Dataset**: MNIST (handwritten digits, 28x28 grayscale images)
- **Latent Dimension**: 2 (for visualization)
- **Architecture**:
  - Convolutional Encoder
  - Latent Sampling with Reparameterization Trick
  - Convolutional Decoder (using Conv2DTranspose)
- **Loss Function**:
  - Binary Cross-Entropy (Reconstruction Loss)
  - KL Divergence (Latent Regularization)

---

## 📂 Project Structure

```bash
.
├── main.py                # Main script for training and evaluation
├── models/
│   ├── vae.keras          # Saved VAE model
│   ├── encoder.keras      # Saved encoder
│   └── decoder.keras      # Saved decoder
├── checkpoint.keras       # Best model checkpoint
├── logs/                  # TensorBoard logs
├── README.md              # Project documentation

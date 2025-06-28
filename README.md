
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


✅ Requirements
Python 3.7+

TensorFlow 2.x

NumPy

Matplotlib

SciPy

Install dependencies:

bash
Copy
Edit
pip install tensorflow numpy matplotlib scipy
🚀 How to Run
Train the VAE:

bash
Copy
Edit
python main.py
Outputs:

Training logs and checkpoints

Latent space visualizations

Generated samples from the latent space

🧠 Key Concepts
Variational Autoencoder (VAE)
A VAE learns to:

Compress input images into a latent space (z_mean, z_log_var)

Sample from that space using the reparameterization trick

Reconstruct the original image using a decoder

Loss function:

total_loss = reconstruction_loss + KL_divergence

Sampling Layer
Implements the reparameterization trick:

python
Copy
Edit
z = z_mean + exp(0.5 * z_log_var) * ε
Where ε ~ N(0, 1).

📊 Visualization
The script generates:

Latent space plots

Sampled points in latent space with decoded images

Reconstruction comparisons (original vs. generated)

💾 Model Saving
After training:

vae.keras: Full VAE model

encoder.keras / decoder.keras: Components saved separately for reuse

📈 TensorBoard
To visualize training:

bash
Copy
Edit
tensorboard --logdir=./logs
📌 Notes
Images are resized to 32x32 with padding for compatibility with conv layers.

The latent space is 2D to enable visualization of the data manifold.

🧪 Future Improvements
Extend to higher-dimensional latent space

Add conditional VAE (cVAE) with label inputs

Apply to more complex datasets (e.g., Fashion MNIST, CIFAR-10)

📧 Contact
For questions or contributions, feel free to open an issue or contact the author.

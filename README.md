# GAN_ASSIGNMENT_4

# GAN Variants on MNIST

This project implements and compares three Generative Adversarial Network (GAN) variants using the MNIST dataset:

- Least Squares GAN (LS-GAN)
- Wasserstein GAN (WGAN)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)

The models are trained, evaluated using quantitative metrics, and visualized with TensorBoard.

---

## 📁 Dataset

We use the classic **MNIST** dataset (handwritten digits) for training and evaluation.

- Downloaded using `torchvision.datasets.MNIST`
- A folder named `original/` contains 100 real MNIST images used for FID comparison and visual inspection

---

## 🧠 Models Implemented

### 1. **LS-GAN**
- Uses Least Squares loss for the discriminator.
- Produces smoother gradients compared to standard GAN.

### 2. **WGAN**
- Uses Wasserstein loss.
- Enforces Lipschitz continuity with weight clipping.

### 3. **WGAN-GP**
- Improved version of WGAN.
- Uses a gradient penalty instead of weight clipping.

---

## 🏋️‍♂️ Training

Each model is trained for **50 epochs** with the following settings:

- Batch Size: 128
- Latent Vector (z): 100-D Gaussian
- Optimizers: Adam (betas tuned per variant)
- Trained on GPU (if available)

---

## 📊 Evaluation

Models are evaluated using:

| Metric             | LS-GAN | WGAN  | WGAN-GP |
|--------------------|--------|-------|----------|
| Inception Score (IS) | 1.00   | 1.02  | **1.69**   |
| FID Score           | 296.95 | 354.99 | **149.36** |

### ✅ Visual Inspection
Generated samples are logged to **TensorBoard** for each GAN variant alongside real MNIST samples.

---

## 🔍 TensorBoard Visualization

Run this command in a Jupyter/Kaggle cell to launch TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir=runs/gan_logs --port=6006

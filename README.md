### 🧠 Autoencoder & VAE Notes
This repository contains foundational notes and practical goals related to Autoencoders (AE) and Variational Autoencoders (VAE). The focus is on understanding core principles, best practices, and applying explainability techniques to these models.
### 🔍 What is an Autoencoder?
An Autoencoder (AE) is a type of neural network used for unsupervised learning. It learns how to compress input data (e.g., images, audio) into a lower-dimensional latent representation and then reconstruct the original input from it.
### 🔧 Components
 - Encoder: Maps the input 𝑥1 to a lower-dimensional latent space 𝑧.
 - Decoder: Reconstructs the original input 𝑥2 from 𝑧.
 - Latent Space (z): Compressed representation that holds the abstract features of the input.
 - Bottleneck Layer: Smaller than the input; forces the model to learn compact and meaningful features.
### 🔁 Objective
𝑥1 → Encoder → 𝑧 → Decoder → 𝑥2

Minimize the reconstruction loss:
```Loss = ∥ 𝑥1 - 𝑥2 ∥```

### 🎯 Representation Learning
Autoencoders aim to learn useful internal representations of the data — features that capture the essence (e.g., shape, style, tone).

### ⚠️ Note: Data should have dependencies across dimensions for effective encoding.

### ⚠️ Note: If the AE uses only linear activations, it becomes mathematically equivalent to Principal Component Analysis (PCA).

### 🧪 Loss Functions
Common choices:

Mean Squared Error (MSE): Good for continuous inputs.

Binary Cross-Entropy (BCE): Suitable for binary or normalized input data.

These measure how closely the reconstruction 𝑥2 matches the original input 𝑥1.

### 🧠 Best Practices
| Component          | Recommended Setup                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Encoder**        | Convolutional layers with **LeakyReLU**, **Batch Normalization**, and **no bias**               |
| **Decoder**        | Convolutional layers with **ReLU**, **Batch Normalization**, and **no bias**                    |
| **Bottleneck**     | Use **Flatten** to reduce dimensions while retaining spatial relationships                      |
| **Decoder Output** | Use **Sigmoid** if image inputs are normalized to **\[0, 1]**, with **BinaryCrossentropy** loss |

### 🧩 Goal of This Work
To:

Build standard Autoencoder and Variational Autoencoder (VAE) models.

Explore and implement explainability techniques for both models.

Potentially investigate other Autoencoder variants for extended analysis.

### 📌 Notes To Be Revised
This document is an initial draft and will be refined over time as more insights are gained through implementation and research.



### 🔍 Important
| Condition                               | Use `model.compile(loss=...)`? | Use `compiled_metrics.update_state()`?            |
| --------------------------------------- | ------------------------------ | ------------------------------------------------- |
| Default model, no custom training loop  | ✅ Yes                          | ✅ Yes                                             |
| Overridden `train_step` (your case)     | ❌ No                           | ❌ Optional, only if you need default metric logic |
| Using custom `VAELoss` class            | ✅ Yes                          | ✅ Yes                                             |
| Using manual loss logic in `train_step` | ❌ No                           | ❌ Optional / custom                               |
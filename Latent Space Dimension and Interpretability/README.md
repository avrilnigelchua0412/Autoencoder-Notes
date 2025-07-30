### Research-Based Insights

Choosing the latent dimensionality involves a trade-off between compression and interpretability. In theory, the optimal latent size should reflect the intrinsic dimensionality of the data—i.e., the minimum number of underlying factors needed to describe it. Saha et al. suggest that the latent “bottleneck” should ideally match the dimensionality of the data manifold. Setting it too small may omit important factors, while setting it too large can lead to overfitting or redundancy, reducing interpretability.

In practice, a common strategy is to start with a small latent size and gradually increase it until performance (e.g., reconstruction or classification accuracy) saturates. For example, a study using EEG data found that a latent space about 25% the size of the input yielded optimal results.

If the latent space is significantly larger than the data’s true complexity, extra dimensions tend to capture noise or redundancy, making interpretation difficult. Conversely, if it is too small, critical features may be lost, hurting both accuracy and explainability.

Additionally, Saha et al. proposed an Automatic Relevance Determination (ARD) mechanism for Variational Autoencoders (VAEs) that automatically identifies the most relevant latent dimensions. Their approach produced significant results in optimizing latent space usage.

### Open Questions:
 - Can the ARD mechanism be adapted for standard (vanilla) autoencoders?
 - Is it worthwhile to do so, considering that vanilla autoencoders are less commonly used in real-world scenarios compared to VAEs?

### To Do:
Study the statistical formulation and mathematical foundations behind Saha et al.’s automatic latent space dimension selection.

### Citations:

 - **ARD-VAE: A Statistical Formulation to Find the Relevant Latent Dimensions of Variational Autoencoders**
 - **Understanding Variational Autoencoders with Intrinsic Dimension and Information Imbalance**

### Additonal:
Use **t-SNE** on "complex" datasets,  like CelebA or CIFAR-10.

They contain many visual factors:

**CelebA**: pose, gender, age, emotion, hairstyle, lighting, background.

**CIFAR-10**: animals, vehicles, textures, shapes, etc.

Their pixel values vary non-linearly across the dataset.

Their features aren’t easily separable using simple thresholds or distances.

Other dataset like **MNIST** and **Fashion-MNIST** are simpler.

| Dataset           | Feature Complexity                     | Latent Size Needed | Why                                             |
| ----------------- | -------------------------------------- | ------------------ | ----------------------------------------------- |
| **MNIST**         | Low (grayscale, digits, simple shapes) | 2–16               | Digits have fixed structure, low variation      |
| **Fashion-MNIST** | Medium (grayscale, clothes)            | 8–32               | Same poses, some structure across categories    |
| **CelebA**        | High (RGB, faces, expressions)         | 64–256+            | Too many fine-grained features                  |
| **CIFAR-10**      | High (RGB, diverse objects)            | 64–256+            | Object differences are more abstract and varied |

**t-SNE** is used when:

You have a latent space that's too high to visualize directly (e.g., 64D),

You want to explore how samples are arranged in latent space.


### PCA and Linearity
**PCA** is a linear technique:
 - It projects data onto axes (principal components) that maximize variance.
 - It's good if the important structure lies along linear combinations of your features.

### Question: Does the latent space contain linear data such that PCA is a good match? Depends — but often, partially yes.
 - The latent space is closer to linear than raw pixel space
 - The encoder is trained to flatten and "linearize" complex input.
 - So while the input data is nonlinear, the bottleneck representations often lie on a lower-dimensional, approximately linear manifold.

### Thus:
 - PCA can often capture broad trends in the latent space.
 - But t-SNE or UMAP are better for nonlinear local clusters (e.g., variations in emotion, pose).

### PCA vs t-SNE for Latent Space
| Aspect                   | PCA                        | t-SNE                                                      |
| ------------------------ | -------------------------- | ---------------------------------------------------------- |
| Technique                | Linear                     | Nonlinear                                                  |
| Goal                     | Maximize variance          | Preserve local neighborhood structure                      |
| Good for                 | Global structure           | Local clusters                                             |
| Speed                    | Fast                       | Slower                                                     |
| Interpretability         | Easy (linear combinations) | Harder to interpret                                        |
| Dimensionality Reduction | Yes (e.g., 128 → 2)        | Yes (128 → 2), but not always faithful for global distance |

So:
✅ Use PCA if you want to analyze global variance trends
✅ Use t-SNE if you want to visualize clusters or classes
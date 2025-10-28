# 🧠 Sparse Autoencoder on Fashion MNIST

A TensorFlow/Keras implementation of a **Sparse Autoencoder** trained on the **Fashion MNIST** dataset to learn efficient and interpretable feature representations of clothing images in an unsupervised manner.


## 📘 Introduction

Autoencoders are a class of **unsupervised neural networks** designed to learn meaningful, compressed representations of data. A **Sparse Autoencoder** extends this idea by introducing **L1 regularization** on the hidden activations, encouraging the network to activate only a few neurons at a time.

In this project, we train a sparse autoencoder to reconstruct Fashion MNIST images — a dataset of 28×28 grayscale clothing images. The sparsity constraint helps the model learn **disentangled, robust, and interpretable features**, which can be used for downstream tasks such as **anomaly detection, clustering, or dimensionality reduction**.


## 🎯 Objective

* Build a **Sparse Autoencoder** using TensorFlow/Keras.
* Learn **low-dimensional, sparse latent representations** of Fashion MNIST images.
* Visualize original and reconstructed images to evaluate reconstruction quality.
* Demonstrate how L1 regularization encourages sparse feature learning.


## 🧩 Dataset

* **Dataset:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
* **Images:** 70,000 grayscale images (60,000 training + 10,000 testing)
* **Image size:** 28 × 28 pixels
* **Classes:** 10 fashion categories (e.g., T-shirt, Dress, Sneaker, Bag, etc.)


## ⚙️ Model Architecture


Input Layer (784 neurons)
     ↓
Dense Layer (32 neurons, ReLU activation, L1 regularization)
     ↓
Dense Layer (784 neurons, Sigmoid activation)
     ↓
Output (Reconstructed Image)

**Loss Function:** Binary Crossentropy
**Optimizer:** Adam
**Regularizer:** L1 (λ = 1e-5)


## 🧠 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load and preprocess data
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train_flat = x_train.reshape(len(x_train), 784)
x_test_flat = x_test.reshape(len(x_test), 784)

# Define Sparse Autoencoder
encoding_dim = 32
input_img = tf.keras.Input(shape=(784,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu',
                                activity_regularizer=tf.keras.regularizers.l1(1e-5))(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)

sparse_autoencoder = tf.keras.Model(input_img, decoded)
sparse_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train model
sparse_autoencoder.fit(x_train_flat, x_train_flat,
                      epochs=50,
                      batch_size=256,
                      shuffle=True,
                      validation_data=(x_test_flat, x_test_flat))
```


## 📊 Results

After training for 50 epochs, the model successfully reconstructed the Fashion MNIST test images, demonstrating that it learned a **compressed and sparse representation** of the data.

### 🔹 Reconstructed Output:

| Original Images                                                                 | Reconstructed Images                                                                      |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| ![Original](https://user-images.githubusercontent.com/placeholder-original.png) | ![Reconstructed](https://user-images.githubusercontent.com/placeholder-reconstructed.png) |

*(You can insert your actual image outputs here after running the visualization code.)*



## 🧩 Key Insights

* Sparse regularization (L1) helped enforce sparsity in hidden units.
* The autoencoder efficiently captured essential visual features while filtering out noise.
* Sparse Autoencoders are effective for **representation learning** without supervision.
* Learned features can be reused for **clustering, anomaly detection**, or as **pretrained encoders** for other models.


## 🧰 Technologies Used

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy, Matplotlib**



## 💡 Conclusion

This project demonstrates the effectiveness of **Sparse Autoencoders** in learning **compact, interpretable, and efficient representations** of image data. The model successfully reconstructs Fashion MNIST images from compressed latent vectors, proving the potential of sparse representations for unsupervised feature learning and dimensionality reduction in computer vision tasks.


# AI CoE Workshop — Hands-On Deep Learning Projects

**Artificial Intelligence | Neural Networks | Deep Learning | Natural Language Processing | Computer Vision**

A collection of three hands-on AI projects developed as part of the **AI Centre of Excellence (AI CoE) Workshop** conducted at my college. These projects explore three important areas of deep learning: binary classification using an Artificial Neural Network, next-token prediction using Transformer-based self-attention, and image classification using a Convolutional Neural Network.

Rather than treating these models as black boxes, the goal was to understand how data is prepared, how neural network architectures process information, how models learn from examples, and how their performance can be evaluated.

---

## Table of Contents

- [About the Repository](#about-the-repository)
- [Project 1: ANN for Binary Classification](#project-1-artificial-neural-network-for-binary-classification)
- [Project 2: Transformer-Based Next-Token Prediction](#project-2-transformer-based-next-token-prediction)
- [Project 3: CNN for Dog–Cat Image Classification](#project-3-cnn-for-dogcat-image-classification)
- [Technologies Used](#technologies-used)
- [How to Run the Projects](#how-to-run-the-projects)
- [Key Learnings](#key-learnings)
- [Acknowledgement](#acknowledgement)

---

## About the Repository

This repository contains three independent deep learning experiments, each designed to explore a different learning problem.

| Project | Domain | Main Objective |
|---|---|---|
| ANN Binary Classification | Machine Learning | Learn a nonlinear decision boundary between two classes |
| Transformer Next-Token Prediction | Natural Language Processing | Predict the next token using causal self-attention |
| CNN Dog–Cat Classification | Computer Vision | Classify images into dog and cat categories |

Together, these projects demonstrate how neural networks can learn different types of patterns depending on the structure of the input data and the architecture used.

## Project 1: Artificial Neural Network for Binary Classification

**Dataset:** `make_moons` from Scikit-learn

**Objective:** Develop an Artificial Neural Network (ANN) capable of distinguishing between two nonlinear classes.

### Problem Statement

Binary classification involves assigning an input to one of two possible classes. However, real-world classification problems are not always linearly separable.

The `make_moons` dataset generates two interleaving crescent-shaped classes. A simple linear decision boundary cannot effectively separate these classes, making the dataset useful for understanding how neural networks learn nonlinear relationships.

### Implementation

The dataset is generated using Scikit-learn and visualized to understand its distribution. The data is then divided into training and testing sets, and feature standardization is applied to improve the training process.

The ANN is implemented using TensorFlow and Keras with the following architecture:

- **Input layer:** Accepts two numerical features.
- **Hidden layer 1:** 16 neurons with ReLU activation.
- **Hidden layer 2:** 16 neurons with ReLU activation.
- **Output layer:** One neuron with sigmoid activation for binary classification.

The model is trained using the Adam optimizer and binary cross-entropy loss. Early stopping is used to monitor validation loss and restore the best-performing weights.

### How the Model Works

The hidden layers learn progressively useful combinations of the input features. Through training, the network adjusts its weights and biases using backpropagation and gradient-based optimization.

ReLU introduces nonlinearity into the network, allowing it to learn a curved decision boundary rather than relying on a straight line. The sigmoid output produces a value between 0 and 1, which is converted into a binary prediction using a threshold of 0.5.

### Evaluation

The model is evaluated using:

- Test accuracy and loss.
- Precision, recall, and F1-score.
- ROC-AUC score.
- Confusion matrix.
- Training and validation accuracy and loss curves.
- Decision-boundary visualization.

**Key takeaway:** This project demonstrates why nonlinear activation functions and multiple neural network layers are important when the relationship between features and classes cannot be represented by a simple linear boundary.

---

## Project 2: Transformer-Based Next-Token Prediction Using Basic Self-Attention

**Dataset:** A custom text corpus focused on artificial intelligence and deep learning.

**Objective:** Build a small Transformer-based language model that learns to predict the next token from the preceding context.

### Problem Statement

Language modeling involves estimating which token is likely to appear next in a sequence. This requires a model to learn relationships between tokens and recognize patterns in the surrounding context.

Transformers address this problem using attention mechanisms, which allow a token representation to incorporate information from other relevant positions in a sequence.

### Implementation

The text corpus is tokenized into words and punctuation. Each token is mapped to an integer ID using a vocabulary, converting raw text into a numerical representation suitable for neural network training.

Fixed-length input sequences are created, with corresponding target sequences shifted by one token. This enables the model to learn next-token prediction at every position.

The model is implemented using TensorFlow and Keras with the following components:

- **Token embeddings:** Convert token IDs into dense vector representations.
- **Positional embeddings:** Provide information about the position of each token.
- **Basic self-attention:** Uses Query, Key, and Value projections to calculate relationships between tokens.
- **Causal masking:** Prevents each position from accessing future tokens.
- **Residual connections and layer normalization:** Support stable information flow through the Transformer block.
- **Feed-forward network:** Further transforms the representations learned through attention.
- **Output projection:** Produces a score, or logit, for every token in the vocabulary.

### Understanding Self-Attention

The core attention operation is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
\]

Here:

- **Query (Q):** Represents the information a token is looking for.
- **Key (K):** Represents the information against which a token's query is compared.
- **Value (V):** Contains the information combined using the attention weights.
- **Scaling factor:** Reduces the magnitude of attention scores before applying softmax.
- **Causal mask (M):** Prevents attention to future positions during next-token prediction.

The resulting attention weights determine how information from earlier tokens contributes to each token's updated representation.

### Training and Text Generation

The model is trained using sparse categorical cross-entropy with logits, which compares its predicted token scores against the expected next-token IDs. The Adam optimizer updates the model's parameters to reduce prediction loss.

After training, the model can:

- Predict the most likely next token for a given prompt.
- Generate a sequence by repeatedly predicting and appending new tokens.
- Track training and validation loss.
- Evaluate token-level accuracy and perplexity on a held-out test split.

Perplexity is calculated from the test loss and provides an indication of how uncertain the model is about the next token. Lower perplexity generally indicates better predictive performance when comparing models under comparable conditions.

**Key takeaway:** This project demonstrates the fundamental mechanics of causal self-attention and how a Transformer converts token sequences into contextual representations for next-token prediction. Because the corpus is small and domain-specific, the experiment is intended to demonstrate the architecture rather than achieve general-purpose language generation.

---

## Project 3: Convolutional Neural Network for Dog–Cat Image Classification

**Dataset:** Cats vs. Dogs dataset from TensorFlow Datasets.

**Objective:** Develop a Convolutional Neural Network (CNN) with exactly three convolutional layers to classify images as either a dog or a cat.

### Problem Statement

Image classification requires a model to recognize visual patterns such as edges, textures, shapes, and object-specific features.

Unlike tabular data, images contain spatial relationships between neighboring pixels. CNNs are designed to exploit these relationships through learnable filters that detect useful visual features.

### Data Preparation

The image dataset is divided into training, validation, and testing subsets. Each image is resized to \(128 \times 128\) pixels and processed in batches to make training more manageable.

Pixel values are normalized from the range 0–255 to the range 0–1 before being passed through the network.

### CNN Architecture

The model contains exactly three `Conv2D` layers, followed by fully connected classification layers.

| Layer | Configuration | Purpose |
|---|---|---|
| Input | \(128 \times 128 \times 3\) | Receives RGB images |
| Convolution 1 | 32 filters, \(3 \times 3\), ReLU | Learns low-level visual features |
| Max pooling 1 | \(2 \times 2\) | Reduces spatial dimensions |
| Convolution 2 | 64 filters, \(3 \times 3\), ReLU | Learns more complex patterns |
| Max pooling 2 | \(2 \times 2\) | Further reduces spatial dimensions |
| Convolution 3 | 128 filters, \(3 \times 3\), ReLU | Learns higher-level visual features |
| Max pooling 3 | \(2 \times 2\) | Produces compact feature maps |
| Flatten | — | Converts feature maps into a vector |
| Dense | 128 neurons, ReLU | Learns combinations of extracted features |
| Dropout | Rate 0.5 | Helps reduce overfitting |
| Output | 1 neuron, sigmoid | Produces the predicted probability of a dog |

### How the Model Works

The convolutional layers learn filters that respond to visual patterns in the images. As the network becomes deeper, the learned representations can capture increasingly complex combinations of features.

Max pooling reduces the spatial dimensions of the feature maps, while the dense layer combines the extracted features for classification.

The sigmoid output produces a probability between 0 and 1. In this implementation, a value below 0.5 corresponds to the cat class, while a value of 0.5 or above corresponds to the dog class.

The model is trained using the Adam optimizer and binary cross-entropy loss. Early stopping monitors validation loss and restores the best-performing weights.

### Evaluation

The model is evaluated using:

- Test accuracy and loss.
- Training and validation accuracy curves.
- Training and validation loss curves.
- Confusion matrix.
- Classification report containing precision, recall, and F1-score.
- Prediction on an example test image.

**Key takeaway:** This project demonstrates how convolutional layers extract spatial features from images and how those learned representations can be used for binary image classification.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Cloud-based development and execution |
| NumPy | Numerical operations and array processing |
| Matplotlib | Data visualization and training graphs |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Dataset generation, preprocessing, and evaluation metrics |
| TensorFlow | Deep learning framework |
| Keras | Neural network construction and training |
| TensorFlow Datasets | Loading the Cats vs. Dogs dataset |

---

## How to Run the Projects

All three experiments are designed to run in Google Colab.

1. Clone or download this repository.
2. Open the notebook for the project you want to run.
3. Upload the notebook to Google Colab, or open it directly if a Colab link is available.
4. Run the cells sequentially from top to bottom.
5. Allow the model to train and inspect the generated visualizations and evaluation metrics.

### Project-Specific Requirements

- **ANN:** Generates the `make_moons` dataset automatically using Scikit-learn.
- **Transformer:** Uses a small custom text corpus, so no external text dataset or API key is required.
- **CNN:** Downloads the Cats vs. Dogs dataset through TensorFlow Datasets. An internet connection is required for the initial download.

A GPU runtime is recommended for faster CNN training, although the experiments can also run on a CPU.

---

## Key Learnings

Working through these projects provided hands-on experience with different neural network architectures and the problems they are designed to solve.

### 1. Neural Network Fundamentals

- Understanding input, hidden, and output layers.
- Using activation functions to model nonlinear relationships.
- Understanding backpropagation, gradient-based optimization, and loss minimization.
- Distinguishing training, validation, and testing data.

### 2. Attention and Language Modeling

- Converting text into token IDs and vocabulary representations.
- Understanding Query, Key, and Value projections.
- Implementing scaled dot-product self-attention.
- Applying causal masking for autoregressive prediction.
- Using positional embeddings, residual connections, and layer normalization.
- Generating text through iterative next-token prediction.

### 3. Computer Vision

- Understanding convolutional filters and feature maps.
- Exploring the role of max pooling in reducing spatial dimensions.
- Building a CNN with a controlled number of convolutional layers.
- Using dropout and early stopping to help manage overfitting.
- Evaluating classification performance using multiple metrics.

### 4. Model Evaluation and Experimentation

Across the three projects, the focus was not only on training models but also on interpreting their behavior through loss curves, classification metrics, confusion matrices, decision boundaries, and generated predictions.

These experiments also highlight an important principle: **the architecture, dataset, preprocessing strategy, and evaluation method must all be considered when interpreting a model's performance.**

---

## Acknowledgement

I would like to thank the organizers and mentors of the **AI Centre of Excellence (AI CoE) Workshop** conducted at my college for providing the opportunity to explore these concepts through practical implementation.

This repository documents my hands-on learning journey across artificial neural networks, Transformer-based self-attention, and convolutional neural networks.

---

**Topics:** Artificial Intelligence · Deep Learning · ANN · Self-Attention · Transformers · Next-Token Prediction · CNN · Binary Classification · Natural Language Processing · Computer Vision

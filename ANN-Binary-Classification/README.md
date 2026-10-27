# Artificial Neural Network for Binary Classification

## Overview

This project demonstrates the implementation of an Artificial Neural Network (ANN) for solving a binary classification problem. The model learns to distinguish between two classes using a nonlinear dataset and demonstrates how neural networks identify complex decision boundaries.

The project was developed as part of the AI CoE workshop to understand the fundamentals of deep learning, model training, and classification.

## Objectives

- Understand the architecture and working principles of an Artificial Neural Network.
- Implement a feedforward neural network using TensorFlow and Keras.
- Explore activation functions, loss functions, and optimization algorithms.
- Train and evaluate a binary classification model.
- Visualize the model's decision boundary to understand its predictions.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow and Keras

## Dataset

The project uses the `make_moons` dataset from Scikit-learn, a synthetic dataset containing two interleaving crescent-shaped classes.

This dataset is useful for demonstrating nonlinear classification because a simple linear decision boundary cannot effectively separate the two classes.

## Model Architecture

The ANN is implemented as a feedforward neural network with the following architecture:

| Layer | Configuration |
|---|---|
| Input layer | 2 input features |
| Hidden layer 1 | 16 neurons, ReLU activation |
| Hidden layer 2 | 16 neurons, ReLU activation |
| Output layer | 1 neuron, Sigmoid activation |

### Key Concepts

**1. Forward propagation**

Input features pass through the hidden layers, where the network learns representations and produces a probability for the positive class.

**2. ReLU activation**

The Rectified Linear Unit introduces nonlinearity in the hidden layers, allowing the network to learn complex patterns.

**3. Sigmoid activation**

The output layer uses the sigmoid function to produce a value between 0 and 1, which can be interpreted as the predicted probability of belonging to the positive class.

**4. Binary cross-entropy**

Binary cross-entropy measures the difference between the true labels and predicted probabilities.

**5. Adam optimizer**

The Adam optimizer updates the network's weights and biases to minimize the loss function during training.

**6. Early stopping**

Early stopping can monitor validation performance and stop training when further training no longer improves the monitored metric.

## Implementation Workflow

1. Generate the nonlinear dataset.
2. Split the data into training and testing sets.
3. Define the ANN architecture using Keras.
4. Compile the model with the Adam optimizer and binary cross-entropy loss.
5. Train the model and monitor its learning progress.
6. Evaluate the model using classification metrics.
7. Visualize the decision boundary to understand how the model separates the two classes.

## Evaluation

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation loss
- Decision boundary visualization

Actual evaluation results should be taken from the notebook's final execution.

## Key Learnings

- How artificial neurons and hidden layers work together.
- Why nonlinear activation functions are important.
- How loss functions and optimizers guide model training.
- How to evaluate a binary classification model.
- How to visualize and interpret a neural network's decision boundary.

## How to Run

1. Open `ANN_Binary_Classification.ipynb` in Google Colab or Jupyter Notebook.
2. Execute the cells in order.
3. Review the training results, evaluation metrics, and decision boundary visualization.

## Project Structure

```text
ANN-Binary-Classification/
├── ANN_Binary_Classification.ipynb
└── README.md
```

## Conclusion

This project provides a practical introduction to neural networks and demonstrates how an ANN can learn nonlinear patterns for binary classification. It establishes a foundation for exploring deeper neural architectures and more complex machine learning problems.

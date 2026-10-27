# Dog and Cat Image Classification Using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to classify images into two categories: dogs and cats.

The model learns visual features directly from image data through convolution and pooling operations, then uses the learned representations to predict the class of an input image.

The project was developed as part of the AI CoE workshop to explore deep learning for image classification and understand the role of CNNs in computer vision.

## Objectives

- Understand the fundamentals of convolutional neural networks.
- Load and preprocess an image dataset for deep learning.
- Implement a CNN using TensorFlow and Keras.
- Train a binary image classification model.
- Evaluate classification performance using suitable metrics.
- Analyze predictions using a confusion matrix and classification report.

## Technologies Used

- Python
- TensorFlow and Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Scikit-learn

## Dataset

The project uses the Cats vs Dogs dataset available through TensorFlow Datasets.

The dataset contains labeled images of cats and dogs. Images are resized to a consistent resolution and prepared for input into the neural network.

### Preprocessing

- Load images and their corresponding labels.
- Resize images to 128 × 128 pixels.
- Prepare image tensors and labels for model training.
- Split or use the dataset's provided training and validation partitions as implemented in the notebook.
- Apply any normalization or rescaling performed by the notebook.

## CNN Architecture

The model uses three convolutional blocks followed by fully connected classification layers.

| Layer | Configuration |
|---|---|
| Input | 128 × 128 RGB image |
| Convolutional layer 1 | 32 filters, ReLU activation |
| Max pooling 1 | Spatial downsampling |
| Convolutional layer 2 | 64 filters, ReLU activation |
| Max pooling 2 | Spatial downsampling |
| Convolutional layer 3 | 128 filters, ReLU activation |
| Max pooling 3 | Spatial downsampling |
| Flatten | Converts feature maps into a vector |
| Dense layer | 128 neurons, ReLU activation |
| Dropout | 0.5 dropout rate |
| Output layer | 1 neuron, Sigmoid activation |

## Key Concepts

### 1. Convolutional Layers

Convolutional layers apply learnable filters to image regions to identify visual patterns such as edges, textures, shapes, and more complex features.

As the network deepens, it can learn increasingly useful representations for distinguishing cats from dogs.

### 2. ReLU Activation

ReLU introduces nonlinearity, allowing the network to learn complex relationships in image data.

### 3. Max Pooling

Max pooling reduces the spatial dimensions of feature maps while retaining prominent activations. This reduces computational requirements and helps build more compact feature representations.

### 4. Flatten Layer

The flatten operation converts the extracted feature maps into a one-dimensional vector that can be passed to the dense classification layer.

### 5. Dropout

Dropout randomly disables a fraction of activations during training. A dropout rate of 0.5 helps reduce overfitting by discouraging excessive reliance on individual neurons.

### 6. Sigmoid Output

The sigmoid output produces a value between 0 and 1, which can be interpreted as the predicted probability for one of the two classes, according to the label encoding.

## Training Process

1. Load the Cats vs Dogs dataset.
2. Resize and preprocess the images.
3. Build the CNN architecture.
4. Compile the model using the Adam optimizer and binary cross-entropy loss.
5. Train the network using labeled images.
6. Monitor training and validation performance.
7. Apply early stopping if configured in the notebook.
8. Evaluate the trained model on the designated evaluation dataset.
9. Generate a confusion matrix and classification report.

## Evaluation

The model's performance can be analyzed using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation loss

The confusion matrix helps identify correct predictions and the types of classification errors made by the model.

Actual evaluation metrics should be taken from the notebook's final executed results.

## Key Learnings

- How CNNs extract meaningful features from images.
- Why convolution and pooling are central to image classification.
- How fully connected layers convert extracted features into predictions.
- How dropout and early stopping can help control overfitting.
- How to evaluate a binary image classification model.
- How to interpret a confusion matrix and classification report.

## Limitations

- The model is designed for a two-class classification task.
- Performance depends on dataset quality, class balance, preprocessing, and training configuration.
- Images that differ significantly from the training data may be misclassified.
- The model is not intended to identify individual animals or classify arbitrary animal species.

## How to Run

1. Open `CNN_Dog_Cat_Classification.ipynb` in Google Colab or Jupyter Notebook.
2. Ensure the required Python packages and dataset are available.
3. Execute the notebook cells in order.
4. Review the training curves and final evaluation results.

## Project Structure

```text
CNN-Dog-Cat-Classification/
├── CNN_Dog_Cat_Classification.ipynb
└── README.md
```

## Conclusion

This project demonstrates how a CNN can learn discriminative visual features to solve an image classification problem. It provides practical experience with image preprocessing, convolutional architectures, model training, regularization, and classification evaluation.

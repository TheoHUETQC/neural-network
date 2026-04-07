# Introduction to Neural Networks in Python

This project is a hands-on introduction to artificial neural networks in Python. Its goal is to help me become familiar with the basic concepts of machine learning through a simple example of binary classification.

## Project Objective

The objective is to build from scratch and train a **neural network** to classify points in a 2D space. The network will learn to distinguish between two classes of points (red and blue) using a Multi-Layer Perceptron (MLP) model.

## Concepts Covered
- Perceptrons and artificial neurons
- Activation functions (sigmoid, ReLU, tanh)
- Training using gradient descent and backpropagation
- Evaluation and visualization of results (loss, accuracy)

## Project Plan

1. **Data Generation**: Creation of a simple dataset with 2D points.
2. **Neural Network Construction**: An MLP with one or two hidden layers.
3. **Model training**: Optimization using gradient descent.
4. **Visualization of results**: Graphical representation of classification and training.
5. **Experiments**: Hyperparameter tuning and testing on other datasets.

## The Project

To view and review the results I obtained, simply open the notebook `neural-network/multi-layer-perceptron/mlp_test.ipynb`. There you will find visualizations of the training process, the results of our neural network, and the neural network along with its biases and all its weights.

## Folder Structure

```
📂 neural-network/
│
├── 📂 perceptron/
│   ├── perceptron.py         ← Perceptron class
│   ├── utils.py              ← Utility functions (visualization, metrics, etc.)
│   ├── main.py               ← Main script (data + training + display)
│   └── perceptron-animation-evolution.py    ← Evolution of a perceptron's learning
│
├── 📂 multi-layer-perceptron/
│   ├── layer.py              ← Layer class for what happens within a layer
│   ├── mlp.py                ← Class for the entire network
│   ├── utils.py              ← Auxiliary functions (visualization, metrics...)
│   └── mlp_test.ipynb        ← Main script (data + training + display)
│
└── 📜 README.md              ← (this file)
```
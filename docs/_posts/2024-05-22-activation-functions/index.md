---
layout: post
title: "Activation Functions"
date: 2024-05-22
published: true
description: "Sigmoid, Tanh, ReLU, Leaky ReLU, and ELU."
---

Activation functions are crucial components of neural networks that introduce non-linearity, enabling networks to learn complex patterns. In this post, we'll explore different activation functions, their properties, and when to use each one.

## Table of Contents

- [What are Activation Functions?](#what-are-activation-functions)
- [Common Activation Functions](#common-activation-functions)
- [Properties and Trade-offs](#properties-and-trade-offs)
- [Implementation Examples](#implementation-examples)
- [Best Practices](#best-practices)
- [Conclusion](#conclusion)

## What are Activation Functions?

Activation functions determine the output of a neural network node given a set of inputs. They introduce non-linearity into the network, allowing it to learn complex patterns and relationships in the data. Without activation functions, neural networks would be limited to learning linear relationships.

## Common Activation Functions

### 1. Sigmoid
The sigmoid function maps any input to a value between 0 and 1:
\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### 2. Tanh
The hyperbolic tangent function maps inputs to values between -1 and 1:
\[ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \]

```python
def tanh(x):
    return np.tanh(x)
```

### 3. ReLU (Rectified Linear Unit)
ReLU is the most commonly used activation function in modern neural networks:
\[ \text{ReLU}(x) = \max(0, x) \]

```python
def relu(x):
    return np.maximum(0, x)
```

### 4. Leaky ReLU
A variant of ReLU that allows small negative values:
\[ \text{LeakyReLU}(x) = \max(\alpha x, x) \]

```python
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)
```

### 5. ELU (Exponential Linear Unit)
ELU helps to push mean activations closer to zero:
\[ \text{ELU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases} \]

```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

## Properties and Trade-offs

Each activation function has its own characteristics:

1. **Sigmoid**
   - Pros: Smooth, bounded output
   - Cons: Suffers from vanishing gradients, not zero-centered

2. **Tanh**
   - Pros: Zero-centered, bounded output
   - Cons: Still suffers from vanishing gradients

3. **ReLU**
   - Pros: Computationally efficient, helps with vanishing gradients
   - Cons: Can suffer from "dying ReLU" problem

4. **Leaky ReLU**
   - Pros: Prevents dying ReLU problem
   - Cons: Introduces an additional hyperparameter

5. **ELU**
   - Pros: Smooth, helps with vanishing gradients
   - Cons: More computationally expensive

## Implementation Examples

Here's how to implement these activation functions in PyTorch:

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()  # or nn.Sigmoid(), nn.Tanh(), etc.
        self.layer2 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

## Best Practices

1. **Hidden Layers**
   - ReLU is the default choice for hidden layers
   - Leaky ReLU or ELU can be used if dying ReLU is a concern

2. **Output Layer**
   - Binary classification: Sigmoid
   - Multi-class classification: Softmax
   - Regression: Linear (no activation) or ReLU for positive outputs

3. **Initialization**
   - Use appropriate weight initialization based on the activation function
   - He initialization for ReLU
   - Xavier/Glorot initialization for sigmoid/tanh

## Conclusion

Choosing the right activation function is crucial for neural network performance. While ReLU has become the default choice for many applications, understanding the properties and trade-offs of different activation functions helps in making informed decisions for specific use cases.

## References

1. [Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.](http://proceedings.mlr.press/v9/glorot10a.html)

2. [He, K., et al. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.](https://arxiv.org/abs/1502.01852)

3. [Clevert, D. A., et al. (2015). Fast and accurate deep network learning by exponential linear units (ELUs).](https://arxiv.org/abs/1511.07289) 
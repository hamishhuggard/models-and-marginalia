---
layout: post
title: "Optimizers"
date: 2024-05-20
published: true
description: "Algorithms for finding the minimum of a loss landscape."
---

Once we have defined the architecture of a model, and thus specified a space of parameter values, we next need an effective way of exploring the model space for a good model.

There is a family of algorithms called optimizers that are used to traverse the parameter landscape and converge towards a minimum. Here's the family tree for the optimizers:

![Optimizer Family Tree]({{ site.baseurl }}/assets/images/optimizers/index.png)

## Stochastic Gradient Descent (SGD)

The most elementary optimizer is stochastic gradient descent (SGD). The idea of SGD is simple: iteratively take steps downhill in the direction of the steepest slope, until you arrive at the bottom of the valley. 

![Gradient Descent]({{ site.baseurl }}/assets/images/optimizers/gradient_descent.png)

This is expressed mathematically as:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)
$$

Where $\theta_t$ is the parameter vector at time step $t$, $\eta$ is the learning rate, and $\nabla_\theta J(\theta_t)$ is the gradient of the loss function with respect to the parameters.

## Momentum

One issue with SGD is that if there is a "gutter" in the loss landscape, the optimizer will bounce back and forth between the two sides of the gutter, which is note very efficient.

![Gutter]({{ site.baseurl }}/assets/images/optimizers/gutter.png)

Momentum is a simple fix to this problem. Again, the idea is simple: give the optimizer momentum analagous to physics. Imagine a ball rolling down a gutter: it will bounce between the sides of the gutter, but its momentum will dampen the oscillations and it will eventually settle down to a more direct path towards the minimum.

![Momentum]({{ site.baseurl }}/assets/images/optimizers/momentum.png)

This is expressed mathematically as:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t) + \gamma \theta_{t-1}
$$

Where $\gamma$ is the momentum coefficient.

## RMSprop

RMSprop is a variant of SGD that adapts the learning rate for each parameter based on the historical gradients. The idea is that we should take larger steps for parameters that are infrequent, and smaller steps for parameters that are frequent.





![Adagrad]({{ site.baseurl }}/assets/images/optimizers/adagrad.png)
![Adam]({{ site.baseurl }}/assets/images/optimizers/adam.png)
![AdamW]({{ site.baseurl }}/assets/images/optimizers/adamw.png)
![Momentum]({{ site.baseurl }}/assets/images/optimizers/momentum.png)
![RMSprop]({{ site.baseurl }}/assets/images/optimizers/rmsprop.png)
![SGD]({{ site.baseurl }}/assets/images/optimizers/sgd.png)

A modern LLM has a search space of trillions of real-valued parameters, and each model call requires trillions of floating point operations. Somehow in this vast search space, we need to find the best combination of parameters

The search space of deep learning is vast, and evaluating loss is expensive. As of 2025, the cutting edge of LLMs have trillions of parameters, which means we need to find the point in a 

Optimizers are algoirthms for efficiently finding a minimum of a loss landscape 

Optimization algorithms are the engines that drive the training of neural networks. They determine how the model's parameters are updated to minimize the loss function. In this post, we'll explore the most important optimizers used in deep learning today, from the foundational [stochastic approximation methods](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full) to modern adaptive optimizers.

![Loss Landscape Visualization](assets/images/optimizers/index.png)
*The complex loss landscape that optimizers must navigate. Different optimizers take different paths to find the minimum.*

## Table of Contents

- [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
- [Adagrad](#adagrad)
- [Momentum](#momentum)
- [RMSprop](#rmsprop)
- [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation)
- [AdamW](#adamw)
- [Choosing the Right Optimizer](#choosing-the-right-optimizer)
- [Best Practices](#best-practices)
- [Code Example](#code-example)
- [Conclusion](#conclusion)

## Stochastic Gradient Descent (SGD)

The most basic optimizer, SGD updates parameters in the direction of the negative gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

Where:
- $\theta_t$ is the parameter vector at time step $t$
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta_t)$ is the gradient of the loss function with respect to the parameters

While simple, vanilla SGD has several limitations:
- It can get stuck in local minima
- It's sensitive to the learning rate
- It doesn't account for parameter-specific learning rates

![SGD Convergence Path](assets/images/optimizers/gradient_descent.png)
*SGD follows the steepest descent path, which can lead to zigzagging in narrow valleys and getting stuck in local minima.*

## Adagrad

[Adagrad](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) (Adaptive Gradient Algorithm) was one of the first optimizers to introduce adaptive learning rates for each parameter. It adapts the learning rate based on the historical gradients:

$$G_{t} = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

Where:
- $G_t$ is the sum of squared gradients up to time step $t$
- $\eta$ is the initial learning rate
- $\epsilon$ is a small constant for numerical stability
- $\odot$ represents element-wise multiplication

Key characteristics:
- Automatically adapts learning rates for each parameter
- Performs larger updates for infrequent parameters
- Performs smaller updates for frequent parameters

Limitations:
- Learning rate can become too small over time
- Accumulation of squared gradients can lead to premature convergence
- Memory requirements grow with the number of parameters

![Adagrad Learning Rate Adaptation](assets/images/optimizers/adagrad.png)
*Adagrad automatically reduces learning rates for frequently updated parameters, leading to more stable convergence.*

## Momentum

[Momentum](https://link.springer.com/article/10.1007/BF01086565) helps SGD overcome local minima by adding a velocity term:

$$v_{t+1} = \gamma v_t - \eta \nabla_\theta J(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

Where:
- $v_t$ is the velocity vector at time step $t$
- $\gamma$ is the momentum coefficient (typically 0.9)

This helps the optimizer:
- Build up speed in directions of consistent gradients
- Overcome small local minima
- Reduce oscillations in narrow valleys

![Momentum vs SGD](assets/images/optimizers/momentum.png)
*Momentum helps the optimizer build up speed in consistent directions and overcome small local minima that would trap vanilla SGD.*

## RMSprop

[RMSprop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) adapts the learning rate for each parameter by dividing the gradient by a running average of its magnitude:

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$$

Where:
- $E[g^2]_t$ is the running average of squared gradients
- $\beta$ is the decay rate (typically 0.99)
- $\epsilon$ is a small constant for numerical stability
- $\odot$ represents element-wise multiplication

Key benefits:
- Adapts learning rates per parameter
- Works well with non-stationary objectives
- Handles different scales of gradients

![RMSprop Adaptive Learning](assets/images/optimizers/rmsprop.png)
*RMSprop adapts learning rates based on recent gradient magnitudes, preventing the learning rate from becoming too small over time.*

## Adam (Adaptive Moment Estimation)

[Adam](https://arxiv.org/abs/1412.6980) combines the benefits of momentum and RMSprop:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

Where:
- $m_t$ is the first moment (mean) of the gradients
- $v_t$ is the second moment (variance) of the gradients
- $\beta_1$ and $\beta_2$ are decay rates (typically 0.9 and 0.999)
- $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected moments

Adam's advantages:
- Combines momentum and adaptive learning rates
- Works well with sparse gradients
- Requires little tuning
- Generally converges faster than other optimizers

![Adam Convergence Comparison](assets/images/optimizers/adam.png)
*Adam typically converges faster than other optimizers by combining momentum with adaptive learning rates.*

## AdamW

[AdamW](https://arxiv.org/abs/1711.05101) is a variant of Adam that implements weight decay correctly:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t - \lambda \theta_t$$

Where:
- $\lambda$ is the weight decay coefficient
- The weight decay term is decoupled from the gradient update

Benefits:
- Better generalization
- More effective weight decay
- Often outperforms Adam in practice

*AdamW often achieves better generalization by implementing weight decay correctly, separate from the gradient update.*

## Choosing the Right Optimizer

When selecting an optimizer, consider:

1. **Problem Type**
   - Adam works well for most problems
   - SGD with momentum can be better for generalization
   - AdamW is preferred when using weight decay

2. **Dataset Size**
   - Large datasets: Adam or AdamW
   - Small datasets: SGD with momentum might generalize better

3. **Model Architecture**
   - Transformers: Adam or AdamW
   - CNNs: SGD with momentum or Adam
   - RNNs: Adam or RMSprop

*Different optimizers perform better on different types of problems and datasets.*

## Best Practices

1. **Learning Rate**
   - Start with the default learning rate
   - Use learning rate scheduling
   - Consider warmup for Adam

2. **Hyperparameters**
   - Adam: $\beta_1=0.9$, $\beta_2=0.999$
   - Momentum: $\gamma=0.9$
   - RMSprop: $\beta=0.99$

3. **Monitoring**
   - Watch for signs of divergence
   - Monitor gradient norms
   - Check parameter updates

*Proper learning rate scheduling can significantly improve convergence and final performance.*

## Code Example

Here's a simple example using PyTorch:

```python
import torch
import torch.optim as optim

# Create a model
model = YourModel()

# Choose an optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

## Conclusion

While Adam and its variants are popular choices, there's no one-size-fits-all optimizer. The best choice depends on your specific problem, dataset, and model architecture. For a more detailed comparison of optimization algorithms, see [Ruder's comprehensive overview](https://arxiv.org/abs/1609.04747).

Remember that the optimizer is just one part of the training process. Proper initialization, learning rate scheduling, and regularization are equally important for successful model training.

*Different optimizers take different paths to convergence, each with their own advantages and trade-offs.*

## References

1. [Robbins, H., & Monro, S. (1951). A Stochastic Approximation Method.](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full) - The original paper introducing stochastic approximation methods.

2. [Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods.](https://link.springer.com/article/10.1007/BF01086565) - Introduces the concept of momentum in optimization.

3. [Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude.](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) - The original RMSprop algorithm.

4. [Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.](https://arxiv.org/abs/1412.6980) - The paper introducing the Adam optimizer.

5. [Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization.](https://arxiv.org/abs/1711.05101) - Introduces AdamW and explains the importance of proper weight decay implementation.

6. [Ruder, S. (2016). An overview of gradient descent optimization algorithms.](https://arxiv.org/abs/1609.04747) - A comprehensive overview of optimization algorithms in deep learning. 
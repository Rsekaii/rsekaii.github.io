---
layout: post
title: "Gaussian Processes - From Concept to Implementation"
date: 2025-02-07
math: true
---



## Introduction

I noticed that writing about my academic journey helps solidify my understanding and provides my future self with a well-established reference whenever I need to recall something.

Thus, whenever I learn something interesting, I will document it here. Whether it will be directly related to my Master's research will be decided later. 

Moreover, I will also include concepts that are related but that I do not fully understand. This paper will help me understand them thoroughly—or so I hope.

---

## Gaussian Processes

We typically assume a model of the form:

$$
y = w_0 + w_1 x + \text{noise}
$$

where \( w_0, w_1 \) are parameters to learn.

A **Gaussian Process (GP)** makes no such assumption—it is non-parametric and assumes a distribution over functions. That is:

- We assume the true function lies in an infinite space of functions.
- We constrain this space using training data.
- Then we assume a **multivariate Gaussian distribution** over any finite collection of function values.

> *Note: It's not the function itself that's Gaussian, but any finite collection of its values is.*

---

### How Do We Constrain the Space of Functions?

Using **covariance** between function values at different training points, typically defined through a **kernel function** like the **RBF kernel**.

---

### The Power of Normality

Given a test point \( x_* \), we can infer the most probable function value \( f(x_*) \) using the normality assumption.

---

### Why Is the Function Gaussian?

Enter **Bayesian Modeling**.

#### Likelihood:

$$
p(D_n \mid w) = \prod_{i=1}^{n} p(y_i \mid x_i, w)
$$

#### Prior:

$$
w \sim p(w)
$$

#### Posterior:

$$
p(w \mid D_n)
$$

#### Predictive Distribution:

$$
p(y \mid x, D_n) = \int p(y \mid x, w) p(w \mid D_n) \, dw
$$

---

### The GP Prior

$$
f(x) \sim \mathcal{N}(m(x), K(x, x'))
$$

- \( m(x) \): mean function  
- \( K(x, x') \): kernel (covariance function)

---

### Likelihood:

$$
y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

---

### GP Prediction:

Assuming:

$$
\begin{bmatrix} f(X) \\ f(x_*) \end{bmatrix} \sim \mathcal{N}
\left( 0, 
\begin{bmatrix} K(X, X) & K(X, x_*) \\
K(x_*, X) & K(x_*, x_*) \end{bmatrix} \right)
$$

Then:

$$
f(x_*) | X, y, x_* \sim \mathcal{N}(\mu_*, \sigma_*^2)
$$

where:

$$
\mu_* = K(x_*, X) [K(X, X) + \sigma_n^2 I]^{-1} y
$$

$$
\sigma_*^2 = K(x_*, x_*) - K(x_*, X) [K(X, X) + \sigma_n^2 I]^{-1} K(X, x_*)
$$

---

## Why Gaussianity?

### Central Limit Theorem (CLT)

The GP prior is a generalization of CLT: the sum of many independent RVs tends toward Gaussianity.

### Gaussian Likelihood:

$$
p(y_i \mid f(x_i)) = \mathcal{N}(f(x_i), \sigma^2)
$$

---

### Properties of Gaussian Distributions

- Any subset of a joint Gaussian is Gaussian.
- Conditional distributions of Gaussians are Gaussian.
- Prior × Likelihood = Posterior is still Gaussian.

---

## Application in Python

We use the [`GPy`](https://sheffieldml.github.io/GPy/) library.

### Data:

```python
N = 10
X = np.linspace(0.05, 0.95, N).reshape(-1, 1)
Y = -np.cos(np.pi * X) + np.sin(4 * np.pi * X) + \
    np.random.normal(loc=0.0, scale=0.1, size=(N, 1))
```

### Kernel Setup

RBF Kernel:

$$
k(x, y) = \sigma^2 \exp\left ( -\dfrac{\|x - y\|^2}{2l^2} \right )
$$

```python
input_dim = 1
variance = 1
lengthscale = 0.2
kernel = GPy.kern.RBF(input_dim, variance=variance, lengthscale=lengthscale)
model = GPy.models.GPRegression(X, Y, kernel)
```

---

## Kernel Parameter Optimization

### Mathematical Form:

$$
k(x, x') = \sigma^2 \exp \left( - \frac{\|x - x'\|^2}{2 l^2} \right)
$$

- \( \sigma^2 \): overall scale (variance)  
- \( l \): how quickly the function varies

![lengthscale impact]({{ "/images/lengthscale-covariance.png" | relative_url }})

---

### GPy Optimization

```python
model = GPy.models.GPRegression(X, Y, kernel)
model.optimize()
```

---

### Noise as Regularization

```python
N = 40
X = np.linspace(0.05, 0.95, N).reshape(-1, 1)
Y = -np.cos(np.pi * X) + np.sin(4 * np.pi * X) + \
    np.random.normal(loc=0.0, scale=0.5, size=(N, 1))

model = GPy.models.GPRegression(X, Y, kernel)
model.optimize()

model.Gaussian_noise.variance.fix(0.01)
model.optimize()
```

**Comparison of high vs low noise:**

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/highvar.png' | relative_url }}" alt="High noise" width="45%">
  <img src="{{ '/images/lowvar.png' | relative_url }}" alt="Low noise" width="45%">
</div>

---

## RBF Limitations

### Complex function:

![Complex function]({{ "/images/complexK.png" | relative_url }})

### Oversmoothing:

![Oversmooth]({{ "/images/RBF2.png" | relative_url }})

### Oscillation:

![Oscillating]({{ "/images/RBF1.png" | relative_url }})


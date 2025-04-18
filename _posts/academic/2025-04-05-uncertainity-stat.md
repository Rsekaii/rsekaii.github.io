---
layout: post
title: " Uncertainty and Some Foundations of Statistical Inference"
date: 2025-03-05
math: true
---

## Introduction

This is an attempt to start writing about my research and learning journey throughout my master's in search of my master's thesis topic. 

As instructed and supported by **Doctor**, starting this post will help me develop the habit of writing and formulating everything I learn and come across that may benefit my thesis. 

**Doctor** gave me a number of introductory papers to initiate the process, and with that, I have started.

---

## Uncertainty in Machine Learning

### Aleatoric vs. Epistemic Uncertainty

Aleatoric uncertainty is **irreducible**: it is intrinsic to the quantity we are measuring. No matter how much additional data we collect, it remains.

Epistemic uncertainty is **reducible**: it arises from data/model limitations and can be reduced with more data or better modeling.

---

### Formal Definitions

Let $ x \in X $ be inputs and $ y \in Y $ outputs.

#### Aleatoric Uncertainty:

$$
Y \mid x \sim f_{Y|x}(y \mid x)
$$

#### Epistemic Uncertainty:

Defined as the remaining uncertainty not captured by aleatoric uncertainty. It can be decomposed further into model, data, and estimation uncertainties.

#### Total Uncertainty:

$$
\text{Total Uncertainty} = \text{Aleatoric Uncertainty} + \text{Epistemic Uncertainty}
$$

##  Statistical Knowledge  
*Statistical & Probabilistic Inference in ML*

### Kullback–Leibler Divergence

Measures the "distance" between two distributions \( P \) (true) and \( Q \) (approximation):

$$
\text{KL}(P \parallel Q) = \sum P(x) \log \frac{P(x)}{Q(x)} \quad \text{(discrete)}
$$
$$
\text{KL}(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx \quad \text{(continuous)}
$$

> KL divergence is **not symmetric**, and equals zero only when $ P = Q $.

---

### Fisher Information

Quantifies how much information a sample provides about a parameter.

#### Definitions:

1. **Variance of the Score Function**:

$$
I(\theta) = \mathbb{E} \left[ S(\theta) S(\theta)^T \right]
\quad \text{where } S(\theta) = \frac{\partial}{\partial \theta} \log p(X | \theta)
$$

2. **Negative Expectation of the Hessian**:

$$
I(\theta) = -\mathbb{E} \left[ \frac{\partial^2}{\partial \theta \partial \theta^T} \log p(X | \theta) \right]
$$

---

#### Interpretation:

- **High Fisher Info** → Low uncertainty in $ \hat{\theta} $
- **Low Fisher Info** → High uncertainty

---

### Cramér-Rao Lower Bound:

$$
\text{Var}(\hat{\theta}) \geq I(\theta)^{-1}
$$

The inverse of Fisher Information provides a lower bound on the variance of any unbiased estimator.

---


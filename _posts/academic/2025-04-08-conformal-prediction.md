---
layout: post
title: "Conformal Prediction - Uncertainty Quantification in Practice"
date: 2025-04-08
math: true
---

## Conformal Prediction

Conformal prediction is a methodology for quantifying uncertainty and producing prediction intervals — similar in spirit to confidence intervals.

Before diving into it, it’s important to understand a key idea:

---

### Scoring Rule

A **scoring rule** evaluates how well a predicted probability distribution aligns with an observed value.

This concept is related to, but distinct from:

- **Loss Functions**: Penalize prediction errors during training.
- **Scoring Functions**: Summarize model performance (like accuracy).
- **Scoring Rules**: Evaluate probabilistic forecasts directly.

---

## Understanding Conformal Prediction

The paper "A Tutorial on Conformal Prediction" by the authors Vovk, Gammerman, and Shafer (who were the orignators of the concept)
is a thorough paper for anyone interested, they also gracefull made a supplementary video breaking it down.
However, here I try to explain it in my own words. So don't expect the same level of detail.

---

### Conformal Prediction Pipeline

The pipeline helped make the method intuitive:

1. **Define Uncertainty**  
   We begin by deciding what "uncertainty" we want to quantify from our model.

2. **Score Function**  
   A nonconformity score function is used to quantify how far a predicted value is from the true value. A well-designed score function gives:

   $$
   s(\hat{y}, y)
   $$

   Higher scores → more uncertainty. Lower scores → greater conformity.

   > ⚠️ Note: This “score function” differs from the Fisher score or scoring rule — terminology can be confusing.

3. **Calibration Scores and Quantile Threshold**  
   Using a **calibration set**, we compute scores and determine the quantile threshold:

   $$
   \hat{q} = \frac{(n+1)(1-\alpha)}{n}
   $$

   For instance, \( \alpha = 0.05 \) gives a 95% confidence level.  
   Now \( \hat{q} \) is the threshold such that ~95% of calibration scores fall below it.

4. **Prediction Set**  
   Given a test point, we define the prediction set as:

   $$
   C(X_{\text{test}}) = \{ y \mid s(\hat{y}_{\text{test}}, y) \leq \hat{q} \}
   $$

   This means: with 95% confidence, the true \( y \) is in this prediction set.

---

## Key Takeaways

Conformal prediction is **model-agnostic** and **distribution-free**.

However, success depends on:

- A well-chosen score function
- The assumption that data is exchangeable
--
### What is Exchangeability?

In probability, a sequence of random variables is **exchangeable** if the joint distribution does not change when the order of the variables is shuffled.

Formally:

$$
P(X_1, X_2, \dots, X_n) = P(X_{\pi(1)}, X_{\pi(2)}, \dots, X_{\pi(n)})
$$
for any permutation $ \pi $ of indices.

This is **weaker than the i.i.d. assumption** — all i.i.d. sequences are exchangeable, but not all exchangeable sequences are i.i.d.

**Why it matters:**  
Conformal prediction only requires exchangeability, not full independence, making it more broadly applicable in real-world data where some dependencies exist.



---

## Application: Tumor Segmentation

One practical application shared by the authors involves detecting gut polyps from endoscopy images.

### Procedure Summary

- A subset of the data is used for calibration.
- A **target TPR (True Positive Rate)** of \( 1 - \alpha \) is set.
- FNR is defined as:

$$
\text{FNR} = 1 - \frac{\sum (\text{Predicted Mask} \cap \text{Ground Truth Mask})}{\sum (\text{Ground Truth Mask})}
$$

- Softmax scores are used as uncertainty scores.
- A threshold \( \lambda_{\text{hat}} \) is computed using **Brent’s method**:

$$
\text{FNR}(\lambda) - \left(\frac{n+1}{n} \alpha - \frac{1}{n}\right) = 0
$$

- Applying $ \lambda_{\text{hat}} $ yields segmentation masks that satisfy the desired confidence level.

---

## Final Thoughts

Despite its simplicity, conformal prediction delivers surprisingly powerful results. Its guarantee of coverage, even without assumptions about the model or distribution, makes it one of the most elegant tools in the uncertainty quantification toolbox.


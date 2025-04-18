---
layout: post
title: "Predicting Feed Stream Yields in Gas Processing Plants Using Machine Learning"
date: 2024-09-08
math: true
---

## Introduction

This post is my initial attempt to apply AI and machine learning techniques to a challenge I often encounter in my professional environment. Due to confidentiality constraints, the data presented here is either synthetic or adjusted, but the methodology and modeling approaches remain valid for educational and demonstration purposes.

Gas processing plants process and treat natural gas and its byproducts. They face operational challenges when feed compositions are unmetered or inaccurately estimated, especially when relying on outdated infrastructure or manual planning. The objective here is to predict product yields from various feed streams using machine learning methods like linear regression and neural networks.

---

## Contributions

- Developed ML models (linear regression, random forest, and neural networks) to predict feed yields.
- Applied these models to address real-world production planning challenges.
- Used correlation analysis and neural networks to extract nonlinear relationships between feeds and products.

---

## Methodology

To understand relationships between feed and product volumes, both visualization and correlation analysis were used. Outliers were removed using the Local Outlier Factor (LOF). We validated linearity with Pearson’s Correlation Coefficient and built neural networks initialized with linear regression weights for faster convergence.

---

## Data Visualization and Exploration

### Single-Feed Plants

<div style="display: flex; gap: 1rem;">
  <img src="/images/PlantB_allPlot_notClean.png" alt="Plant B" width="48%">
  <img src="/images/plantK_products.png" alt="Plant K" width="48%">
</div>

### Multi-Feed Plants

<div style="display: flex; gap: 1rem;">
  <img src="/images/PlantX_NGLColor_feeds_notClean.png" alt="Plant X" width="48%">
  <img src="/images/PlantY_iColor_products_notClean.png" alt="Plant Y" width="48%">
</div>

---

## Correlation Matrices

**Plant B Correlation Matrix:**

|          | Feed B | C1     | NGL    |
|----------|--------|--------|--------|
| Feed B   | 1.00   | 0.95   | 0.67   |
| C1       | 0.95   | 1.00   | 0.67   |
| NGL      | 0.67   | 0.67   | 1.00   |

**Plant X Correlation Matrix (truncated):**

|          | Feed B | Feed S | Feed i | C1     | NGL    | Sulphur |
|----------|--------|--------|--------|--------|--------|---------|
| Feed B   | 1.00   | -0.21  | -0.37  | 0.17   | 0.26   | 0.21    |
| Feed S   | -0.21  | 1.00   | -0.39  | 0.31   | -0.11  | 0.46    |
| Feed i   | -0.37  | -0.39  | 1.00   | 0.09   | 0.30   | -0.28   |
| C1       | 0.17   | 0.31   | 0.09   | 1.00   | 0.35   | 0.68    |

---

## Model Evaluation

### Regression Performance (Plant X):

| Model                | R²      | MSE      | RMSE    | MAE     | CMAPE   |
|----------------------|----------|----------|---------|---------|---------|
| Linear Regression    | 0.305    | 137271   | 370.5   | 223.3   | 0.910   |
| Decision Tree        | -0.092   | 178355   | 422.3   | 204.4   | 0.915   |
| Random Forest        | 0.446    | 124168   | 352.4   | 186.1   | 0.929   |
| SVR                  | 0.036    | 217435   | 466.3   | 230.9   | 0.910   |

### Regression Performance (Plant Y):

| Model                | R²      | MSE      | RMSE    | MAE     | CMAPE   |
|----------------------|----------|----------|---------|---------|---------|
| Linear Regression    | 0.403    | 377322   | 614.3   | 331.4   | 0.911   |
| Decision Tree        | 0.350    | 374735   | 612.2   | 309.5   | 0.910   |
| Random Forest        | 0.598    | 212575   | 461.1   | 229.4   | 0.931   |
| SVR                  | 0.062    | 830996   | 911.6   | 486.5   | 0.884   |

---

## Neural Networks

Neural networks improved fit quality significantly. Linear regression weights were used for initialization to improve convergence.

<div style="display: flex; gap: 1rem;">
  <img src="/images/wihwithout.png" alt="With vs Without Weight Init" width="48%">
</div>

**Best NN Evaluation (Plant Y, 5-fold cross-validation):**

- **MSE:** 289,751  
- **CMAPE:** 91.16%  
- **R²:** 0.989

---

## Yield Comparison

| Feed | C1 (Emp) | NGL (Emp) | Sulphur (Emp) | C1 (NN) | NGL (NN) | Sulphur (NN) |
|------|----------|-----------|----------------|---------|----------|---------------|
| B    | 0.72     | 7.80      | 1.55           | 1.18    | 9.74     | 2.69          |
| S    | 0.84     | 2.00      | 2.00           | 0.12    | 9.35     | 0.17          |
| K    | 0.70     | 9.47      | 0.00           | 0.14    | 3.10     | 0.00          |
| i    | 0.84     | 2.49      | 0.90           | 0.42    | 3.86     | 1.43          |
| O    | 0.50     | 23.40     | 1.90           | 0.98    | 12.18    | 2.12          |
| N    | 0.80     | 6.67      | 0.60           | 1.75    | 10.45    | 1.42          |

---

## Conclusion

This study demonstrates how machine learning — especially neural networks — can improve yield estimation accuracy in gas processing. Even with limited features and noisy data, neural models outperformed traditional approaches. With better instrumentation or real-time data, these tools could become core to modern energy operations.


---
layout: post
title: "Causal Effects of Monetary Policy on Unemployment: A Structural Approach"
date: 2025-03-18
math: true
---

## Introduction

As someone currently preparing for the CFA exams, I’ve been diving deeper into how macroeconomic policy impacts inflation, employment, and GDP. The curriculum often presents these causal relationships as established fact — but I became curious: **can we test these causal claims statistically?**

This project is a personal attempt to bridge what I’m learning in economics from my CFA preparations with the tools of **causal inference** I learned in my Master’s in Machine Learning.

In particular, I wanted to see whether central bank interest rate changes truly affect unemployment, and how we can model that effect using real-world data and tools like **do-calculus**.

---

## Data Selection

For this analysis, I used data from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) due to its high-frequency monthly observations. Key indicators include:

- **FEDFUNDS**: Federal Funds Rate
- **M2SL**: M2 Money Supply
- **CPIAUCSL**: Consumer Price Index (Urban, Seasonally Adjusted)
- **GDP Growth**: Real GDP % Change (Quarterly)
- **UNRATE**: Unemployment Rate

To visualize assumed relationships between these variables, I started with a basic DAG inspired by standard macroeconomic models:

<div style="display: flex; justify-content: center;">
  <img src="{{ '/images/DAG_cfa.png' | relative_url }}" alt="Causal DAG" width="600px">
</div>

---

## Data Preprocessing

Since most variables are monthly except GDP (which is quarterly), I interpolated GDP values so each quarter value was copied across the respective three months.

Exploratory time-series plots revealed correlations and suggested directions for further causal analysis:

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/ts1.png' | relative_url }}" alt="Time Series Plot 1" width="48%">
  <img src="{{ '/images/ts2.png' | relative_url }}" alt="Time Series Plot 2" width="48%">
</div>

To prepare the data for time-series modeling, we applied first-order differencing to ensure stationarity. This was verified using the Augmented Dickey-Fuller (ADF) test. This step was crucial to avoid spurious results in downstream Granger causality and VAR models.

<div style="display: flex; justify-content: center;">
  <img src="{{ '/images/ts3.png' | relative_url }}" alt="Transformed Time Series Plot" width="600px">
</div>

---

## Granger Causality and DAG Adjustment

To uncover temporal causality, I ran **Granger Causality** tests between variables. Some findings:

- The Federal Funds Rate Granger-causes both GDP and Inflation.
- GDP Granger-causes M2 and Unemployment at some lags.
- Unemployment Granger-causes Inflation at longer lags.

These empirical results contradicted some textbook claims, prompting me to revise the initial DAG structure to better reflect observed temporal dependencies:

<div style="display: flex; justify-content: center;">
  <img src="{{ '/images/DAG_new.png' | relative_url }}" alt="Updated DAG" width="600px">
</div>

---

## Modeling the Effect of Interest Rate Intervention

The key question:  
What is the effect on **Unemployment** when the **Central Bank** changes interest rates?

Formally:
$$
P(\text{Unemployment\%} \mid do(\text{Fed Fund Rate}))
$$

After applying **do-calculus**, the corresponding DAG is:

<div style="display: flex; justify-content: center;">
  <img src="{{ '/images/DAG_dofed.png' | relative_url }}" alt="DAG with do(FedFunds)" width="600px">
</div>

---

## Deriving the Interventional Distribution

Using the adjusted DAG and applying rules of **do-calculus**, I arrived at:



Key reasoning:
- Conditioning M2 on GDP adjusts for confounders.
- Summing over GDP, Inflation, and M2 marginalizes confounding paths.
- The structure respects endogenous roles of variables like M2 and Inflation.

However, instead of analytically estimating these conditional distributions, I opted for a more flexible and non-parametric approach.

---

## KDE Estimation and Causal Simulation

We estimated the full joint distribution over the variables using **Kernel Density Estimation (KDE)**, then simulated the effect of different interventions on the Federal Funds Rate.

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/kde1.png' | relative_url }}" alt="KDE Plot 1-1" width="48%">
  <img src="{{ '/images/kde2.png' | relative_url }}" alt="KDE 3D Color Plot" width="48%">
</div>

Causal effect plots:

- $$ \mathbb{E}[\text{Unemployment} \mid do(\text{FedFunds})] $$

<div style="display: flex; justify-content: center;">
  <img src="{{ '/images/cp1.png' | relative_url }}" alt="Causal Effect Plot: Unemployment" width="600px">
</div>

- $$ \mathbb{E}[\text{GDP Growth} \mid do(\text{FedFunds})] $$

<div style="display: flex; justify-content: center;">
  <img src="{{ '/images/cp2.png' | relative_url }}" alt="Causal Effect Plot: GDP Growth" width="600px">
</div>

---

## Extra: Reflecting on the Predictive vs Causal Gap

To contrast this causal approach with older data science techniques I previously used, I ran **Recursive Feature Elimination (RFE)** with linear regression to predict unemployment. This method iteratively removes features to find the optimal subset that best predicts the outcome — something I would have once naively interpreted as causal.

Interestingly, **Inflation** and **M2 Money Supply** emerged as the best predictors of Unemployment. However, as shown through the causal analysis above, **neither has a direct causal effect** on Unemployment. M2 acts as a mediator for the Federal Funds Rate, and Inflation may even act as a collider or proxy in certain paths — invalidating causal inference if conditioned on.

This experiment highlights a key lesson: predictive power \( \neq \) causal influence. Variables that are easiest to observe or strongly correlated might still be poor intervention targets. Causal inference provides a fundamentally different lens — one focused not on "what correlates," but on "what drives change."

---

## Final Thoughts

In this project, I successfully:
- Modeled macroeconomic indicators using VAR and KDE
- Transformed data to satisfy stationarity assumptions
- Tested economic theory using Granger causality and DAGs
- Simulated interventional distributions using do-calculus

This end-to-end pipeline demonstrates the power of combining causal reasoning, time-series modeling, and machine learning. It provides a fresh, structured way to analyze classic economic questions through a modern lens.

[View the full code on GitHub](https://github.com/Rsekaii/CausalCFA)

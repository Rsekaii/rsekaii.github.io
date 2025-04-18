---
layout: post
title: "Causal Effects of Monetary Policy on Unemployment: A Structural Approach"
date: 2025-03-18
math: true
---

## Introduction

As someone currently preparing for the **CFA exams**, I’ve been diving deeper into how macroeconomic policy impacts inflation, employment, and GDP. The curriculum often presents these causal relationships as established fact — but I became curious: **can we test these causal claims statistically?**

This project is a personal attempt to bridge what I’m learning in economics from my CFA preparations using the tools of **causal inference** I learned in the Masters in Machine Learning.
In particular, I wanted to see whether Central Bank interest rate changes truly affect unemployment, and how we can model that effect using real-world data and tools like **do-calculus**.

---

## Data Selection

For this analysis, I used data from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) due to its high-frequency monthly observations. Key indicators include:

- **FEDFUNDS**: Federal Funds Rate
- **M2SL**: M2 Money Supply
- **CPIAUCSL**: Consumer Price Index (Urban, Seasonally Adjusted)
- **GDP Growth**: Real GDP % Change (Quarterly)
- **UNRATE**: Unemployment Rate

To visualize assumed relationships between these variables, I started with a basic DAG inspired by standard macroeconomic models:

![Causal DAG](/images/DAG cfa.png)

---

## Data Preprocessing

Since most variables are monthly except GDP (which is quarterly), I interpolated GDP values so each quarter value was copied across the respective three months.

Exploratory time-series plots revealed correlations and suggested directions for further causal analysis:

![Time Series Plot 1](/images/ts1.png)  
![Time Series Plot 2](/images/ts2.png)

---

## Granger Causality and DAG Adjustment

To uncover temporal causality, I ran **Granger Causality** tests between variables. For instance:

- GDP lag of ~3–5 months
- FEDFUNDS and M2 consistently Granger-caused inflation and GDP
- Bidirectional relationships between GDP and unemployment

Based on these results, I updated the DAG:

![Updated DAG](/images/DAG new.png)

---

## Modeling the Effect of Interest Rate Intervention

The key question:  
What is the effect on **Unemployment** when the **Central Bank** changes interest rates?

Formally:
$$
P(\text{Unemployment\%} \mid do(\text{Fed Fund Rate}))
$$

After applying **do-calculus**, the corresponding DAG is:

![do-FED DAG](/images/DAG dofed.png)

This isolates the intervention on interest rates from their usual causes (i.e., cutting all incoming arrows to FEDFUNDS).

---

## Deriving the Interventional Distribution

Using the adjusted DAG and applying rules of **do-calculus**, I arrived at:

$$
P(\text{Unemployment\%} \mid do(\text{Fed Fund Rate})) =
\sum_{\text{Inflation}, \text{M2}, \text{GDP}}
P(\text{Unemployment\%} \mid \text{M2})
P(\text{M2} \mid \text{GDP}, \text{Inflation})
P(\text{GDP} \mid \text{Fed Fund Rate})
P(\text{Inflation} \mid \text{M2})
$$

Key reasoning:
- Conditioning M2 on GDP adjusts for confounders.
- Summing over GDP, Inflation, and M2 marginalizes confounding paths.
- The structure respects endogenous roles of variables like M2 and Inflation.

---

## Next Steps

I plan to estimate these conditional distributions using **Kernel Density Estimation (KDE)** to simulate:
$$
P(\text{Unemployment\%} \mid do(\text{Fed Fund Rate}))
$$

This will help visualize the shift in unemployment distribution resulting from monetary policy intervention.

---

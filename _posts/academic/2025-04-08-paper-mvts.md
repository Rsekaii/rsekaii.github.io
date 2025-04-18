---
title: " Paper Review - Simple Baseline ofr MultiV. T.S. Forecasting"
subtitle: "A Simple Baseline for Multivariate Time-Series Forecasting"
author: Rames AlJneibi
date: 2024-10-16
layout: post
math: true
---

In contrast to the current trend of re-purposing Large-Language Models (LLMs) for Multivariate Time-Series (MTS) forecasting, the paper introduces a novel approach that models MTS using well-known tokenization techniques, enhanced by an innovative generalization of the self-attention mechanism. This approach reduces computational complexity while improving performance.

---

### Tokenization

$$
X(t) \xrightarrow{\text{SWT}} \{A_t^s, D_t^s\}_{s=1}^S
$$

SWT decomposes MTS into approximation and detail coefficients at each scale \( s \).

The Stationary Wavelet Transform (SWT) offers an ideal representation for time-series data as it avoids down-sampling, ensuring robust reconstruction. By utilizing learnable low-pass and high-pass filters \( h_0 \) and \( g_0 \), and up-sampling at every scale, no information is lost in the process. Each token \( u_{t}^{s} \) captures information at a specific scale \( s \) and time \( t \). While redundancy is typically computationally intensive, in this context, it helps reduce the model's parameters compared to LLMs.

---

### Self-Attention Generalization

**Geometric Product**:  
$$
QK = Q \cdot K + (Q \wedge K)
$$


Conventional self-attention relies on the dot product to capture relationships between the \( Q \) and \( K \) matrices. The paper enhances self-attention by introducing the **geometric product**, which complements the dot product with the **wedge product**. This addition captures linear independencies and the orientation of the spanned space. The outputs of both products are summed after matching their dimensions, creating a more robust attention mechanism.

---

### MTS Reconstruction

$$
\begin{array}{c@{\hskip 3em}c}
\hat{X} = \{\hat{x}_1, \dots, \hat{x}_{L'}\} = \hat{a}^{(0)} &
\hat{a}_t^{(s-1)} = \sum_k h_1^{(s)}(k) \hat{a}_{t+k}^{(s)} + \sum_k g_1^{(s)}(k) \hat{u}_{t+k}^{(s)}
\end{array}
$$

The processed tokens are converted back to the time domain using a learnable inverse SWT and filters \( h_1 \) and \( g_1 \). Starting from the widest scale, the model iteratively combines the coefficients to reconstruct the time-series \( \bar{X} \), which is then passed through a feed-forward network to generate the final forecast. The model is trained end-to-end by minimizing the prediction loss via backpropagation.

---

### Conclusion

The proposed model demonstrates a highly efficient design and strong performance compared to existing baselines across diverse MTS datasets. However, its competitive edge is especially notable under low-data regimes. As larger MTS datasets become more available, LLMs may eventually outperform this method due to their scaling advantage.

---

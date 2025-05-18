---
layout: post
title: "Conformal Prediction - Uncertainty Quantification in Practice"
date: 2025-04-08
math: true
---



Conformal prediction (CP) is a method to quantify predictive uncertainty. It gives you prediction sets or intervals that come with guaranteed coverage — even when we don’t know the underlying distribution.

Before diving in, I had to wrap my head around a new concept: the **scoring rule**.

### Scoring Rule

A scoring rule tells us how good our probabilistic predictions are. It compares what the model *believed would happen* versus what *actually happened*.

This concept felt a bit redundant at first — like, isn't this just a loss function? Or maybe accuracy? But I came to understand they serve different goals:

- **Loss Function:** Used during training — it penalizes the difference between predicted and true values.
- **Scoring Rule:** Evaluates probabilistic predictions (like full softmax vectors), measuring how well the predicted distribution matches reality.
- **Scoring Function:** Used more loosely, often referring to evaluation metrics like accuracy or F1.

[Note: While I initially thought the nonconformity score in CP is a scoring rule, I later learned it's not quite the same.
it's just a function used to measure how “strange” a prediction is. It doesn’t have to be proper or differentiable like formal scoring rules.]

### Understanding Conformal Prediction

After reading a few technical papers (and finding a really helpful YouTube explainer — shoutout to the authors!), I’ll try to describe what I understood in my own words.
If I can explain it, I probably understand it..right?

### Conformal Prediction Pipeline

The following step-by-step framework helped me solidify the concept:

1. **Define uncertainty:** Pick a way to measure how “uncertain” a prediction is. This could be low softmax probability for the true class, or any other measure capturing the model's confidence.
2. **Choose a nonconformity score:** This function takes in the model's output and a label, and tells you how “nonconforming” or odd the prediction is. The higher the score, the more uncertain.

   (Yes, it's called a “score,” but it’s unrelated to scoring rules in the probabilistic forecast sense.)

3. **Calibration:** Use a *held-out calibration set* (not used for training) to compute the nonconformity scores. Then compute a quantile of those scores:

   $$
   \hat{q} = \text{Quantile}_{1 - \alpha} \left( \text{calibration scores} \right)
   $$

4. **Prediction sets:** For a test input \\( x_{\text{test}} \\), the model outputs a distribution. We compute the nonconformity score for each possible label \\( y \\), and include all labels such that:

   $$
   C(x_{\text{test}}) = \{ y \mid s(f(x_{\text{test}}), y) \leq \hat{q} \}
   $$

   That’s our prediction set. We can say with \\(1 - \alpha\\) confidence that the true label lies within it.

### Critical Considerations

At first glance, this all feels like magic: you get valid uncertainty estimates, **without needing to model the data distribution**, and it works for any base model.

But there's a catch or two:

- The choice of the nonconformity score matters *a lot*. Even though coverage is guaranteed, the size of the prediction set and its usefulness depends entirely on this score.
- The theoretical guarantees rely on a strong assumption: **exchangeability**.

#### Exchangeability

Exchangeability is a weaker assumption than i.i.d., but it's the key assumption underpinning conformal prediction.
A sequence of random variables \\( Z_1, Z_2, \dots, Z_n \\) is said to be **exchangeable** if their joint distribution remains unchanged under any permutation:

$$
P(Z_1, \dots, Z_n) = P(Z_{\pi(1)}, \dots, Z_{\pi(n)}) \quad \text{for any permutation } \pi
$$

In simpler terms: the order in which data points appear doesn’t matter. All that matters is the set itself.

### Next Steps

Despite these caveats, conformal prediction still feels like the Central Limit Theorem’s cool cousin. It works when it really shouldn’t.

I plan to continue testing conformal prediction in practice: classification, OOD data, different scores .
Hopefully, through these experiments, I’ll gain better intuition and maybe stumble upon a research-worthy insight.



---

## Conformal Prediction with a CNN on MNIST

We begin with a foundational experiment using the well-known MNIST dataset, which consists of grayscale images of handwritten digits (0–9).
The data is originally split into 60,000 training and 10,000 test samples.
I further split the training set into 50,000 samples for training and 10,000 for calibration, which are used exclusively for conformal prediction.

To classify digits, I trained a simple convolutional neural network (CNN). It’s a small one: just two convolutional layers followed by a couple of dense layers.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # Output: 26x26
        x = F.relu(self.conv2(x))       # Output: 24x24
        x = F.max_pool2d(x, 2)          # Output: 12x12
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

I trained this model for five epochs using Adam and cross-entropy loss. Once trained, I used it to generate predictions on both the calibration and test sets.
The outputs are logits, which are turned into probabilities using the softmax function.

To build conformal prediction sets, I used a simple nonconformity score: \\( 1 - p_y \\), where \\( p_y \\) is the model’s probability for the correct class.
Intuitively, this score gets smaller when the model is confident, and larger when it's unsure.

```python
def get_softmax_scores(loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)

```

Then, I calculated the nonconformity scores for the calibration set, and for different confidence levels, I constructed prediction sets using the \\( (1 - \alpha) \\)-quantile.

```python
cal_nonconformity = 1 - np.array([p[y] for p, y in zip(cal_probs, cal_labels)])
def predict_sets(probs, threshold):
    return [np.where(1 - p <= threshold)[0] for p in probs]

alphas = [0.05, 0.1, 0.15]
for alpha in alphas:
    q = np.quantile(cal_nonconformity, 1 - alpha)
    pred_sets = predict_sets(test_probs, q)
    coverage = np.mean([y in s for y, s in zip(test_labels, pred_sets)])
    avg_size = np.mean([len(s) for s in pred_sets])
    print(f"α={alpha:.2f} => Coverage: {coverage:.3f}, Avg Set Size: {avg_size:.2f}")

```
**Example Output:**
<pre><code>
α = 0.05 → Coverage: 0.957, Avg Set Size: 0.96
α = 0.10 → Coverage: 0.910, Avg Set Size: 0.91
α = 0.15 → Coverage: 0.865, Avg Set Size: 0.87
</code></pre>



The average prediction set size is under 1 which makes sense. Most predictions are confident, returning a single label.
A few are highly uncertain and return empty sets. Here’s how many predictions had which set sizes for α = 0.1

<pre><code>
Prediction sets with 0 entries: 892
Prediction sets with 1 entries: 9108
</code></pre>

So far, things are behaving just as expected. Coverage is on target, and set sizes reflect the model’s confidence.


## CIFAR Experiments

Now for something more challenging. I ran the same experiment setup on CIFAR, which is a colorful image classification dataset.
I trained another basic CNN on CIFAR-10, used conformal prediction to estimate uncertainty, 
and then pushed it a bit further by testing on CIFAR-100 — totally out-of-distribution (OOD).

## Softmax Score

similar to the previous one: three-layer CNN, 5 epochs, Adam optimizer. I split CIFAR-10 into 45k for training and 5k for calibration.

I started with the same softmax-based score: $ 1 - p_{\text{true}} $. For $ \alpha = 0.1 $, I got:

<pre><code>
α = 0.10 → Coverage: 0.894, Avg Set Size: 1.76
Prediction sets with 1 entries: 5186
Prediction sets with 2 entries: 2836
Prediction sets with 3 entries: 1329
...
</code></pre>

Clearly, the model is less confident than on MNIST. Here's a plot showing which labels tend to result in larger prediction sets:

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/CIFA10sets.png' | relative_url }}" alt="High noise" width="45%">
</div>
<p style="font-size: 0.9rem; color: #666; margin-top: 0.2rem;">
  Figure: True labels vs prediction set sizes (CIFAR-10).
</p>




Next, I tested on CIFAR-100 — completely different labels — using the same conformal setup:
<pre><code>
OOD (CIFAR-100) → Avg Prediction Set Size: 2.63
Coverage: 0.03
Top-1 probability on CIFAR-100: 0.661 ± 0.209
...
</code></pre>

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/CIFAidodsetss.png' | relative_url }}" alt="High noise" width="45%">
</div>
<p style="font-size: 0.9rem; color: #666; margin-top: 0.2rem;">
  Figure: Prediction set sizes for CIFAR-10 vs CIFAR-100.
</p>


Despite the low coverage (as expected), the model still returns small prediction sets on OOD data. 
First thing that comes to mind is to modify the model to increase its accuracy or maybe include a label that is made for unclear images.

But what I wanted to do is test conformal prediction usefullness when dealing with OOD data, so I'll try another approach. 
The over confidence in predictions sets is an issue of CP implementation, more specifically our nonconformal score may not
be sensitive enough leading to small prediction sets even when the model is wrong.

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/ODDoc1.png' | relative_url }}" alt="High noise" width="45%">
</div>
<p style="font-size: 0.9rem; color: #666; margin-top: 0.2rem;">
  Figure: Examples of when the model was overconfident in its wrong predictions, sets with less than 2 labels using softmax scores 
   (an orange being labeled as either a frog or a dog — “overconfident” is an understatement).
</p>

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/ODDoc2.png' | relative_url }}" alt="High noise" width="45%">
</div>
<p style="font-size: 0.9rem; color: #666; margin-top: 0.2rem;">
  Figure: More examples of overconfidence when using softmax scores, King of the jungle is offended.
</p>

So lets look for a more sensitive score, that should make our prediction sets deal with OOD data with more caution.


## Margin Score

So I tried something more reactive and sensitive to the model’s uncertainty: the \textbf{margin score}. 
It’s defined as the difference between the top-1 and top-2 predicted probabilities. The smaller this margin, the more uncertain the model is assumed to be.
And prediction sets are built by checking how close other classes are to the top-1.

To clarify with an example, suppose we are testing two new images, A and B, and the model returns the following probability distributions:
**Example:**

- Image A: {0.01, 0.90, 0.05, 0.04} → margin = 0.90 - 0.05 = 0.85  
- Image B: {0.20, 0.10, 0.40, 0.30} → margin = 0.40 - 0.30 = 0.10

> *Note: Upon reflection, this is clearly not a valid nonconformity score. But I’m keeping it here for the sake of documenting the progression of my experiments.*

```python
cal_nonconformitym = np.array([
    np.sort(p)[-1] - np.sort(p)[-2]
    for p in cal_probs
])
def predict_sets_margin(probs, threshold):
    sets = []
    for p in probs:
        top1 = np.max(p)
        pred_set = np.where((top1 - p) <= threshold)[0]
        sets.append(pred_set)
    return sets
```

Rsults : 

<pre><code>
CIFAR-10 → Avg Set Size: 8.93
CIFAR-100 → Avg Set Size: 9.94
</code></pre>

So margin score is definitely more cautious since it inflates the set size aggressively, even for in-distribution data.
But it's more sensitive to OOD uncertainty, which we wanted. However, it does not seem worth the price. 



## Entropy Score

While working on the margin score, I thought: instead of comparing just the top two probabilities, why not measure the full uncertainty?
If margin score measures the spread of the model's top two predictions, is there a way to capture the full spread of all output probabilities ? 
Enter entropy.

$$
H(p) = -\sum_i p_i \log(p_i)
$$

Entropy captures how “spread out” the probability distribution of the predicted label is:

- Sharp peaks → model is confident → low entropy  
- Flat predictions → model is uncertain → high entropy

```python
def entropy(p):
    return -np.sum(p * np.log(p + 1e-12))

cal_non_e = np.array([entropy(p) for p in cal_probs])
q_e = np.quantile(cal_non_e, 1 - alpha)
```
After defining entropy, we use it as a conformal score by computing the entropy of each sample in the calibration set.
We then determine the appropriate threshold (quantile) from these calibration scores.

To construct the prediction sets for new samples, we sort the predicted probabilities in descending order.
Starting from the highest, we incrementally include class probabilities one by one, recalculating the cumulative entropy at each step. 
We stop once the entropy exceeds the predefined threshold.

```python
def pred_entropy_thresholded(probs, entropy_thresh):
    sets = []
    for p in probs:
        sorted_indices = np.argsort(p)[::-1]
        cumulative_p = [], cumulative_entropy = 0 , current_set = []
        for idx in sorted_indices:
            pi = p[idx] , cumulative_p.append(pi)
            cumulative_entropy = -np.sum(np.array(cumulative_p) * np.log(np.array(cumulative_p) + 1e-12))
            current_set.append(idx) , if cumulative_entropy > entropy_thresh:
                break
        sets.append(current_set), return sets
```
**Results:**

<pre><code>
CIFAR-10 → Avg Set Size: 9.52
CIFAR-100 → Avg Set Size: 8.78

</code></pre>

Here’s the issue: even for confident predictions, entropy takes a while to “approve” a small set. It accumulates slowly when probabilities are concentrated.
For example, \([0.90, 0.05, ...]\) still requires multiple labels before entropy passes the threshold.

---

## Conclusion

This shows that the choice of nonconformity score really matters. All scores satisfy the conformal guarantee, but they behave very differently in practice.
Especially when it comes to how they respond to model uncertainty.

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/CIFApredid.png' | relative_url }}" alt="High noise" width="45%">
</div>
<p style="font-size: 0.9rem; color: #666; margin-top: 0.2rem;">
  Figure: {Average prediction set size vs \(\alpha\), CIFAR-10 (in-distribution).
</p>

<div style="display: flex; gap: 1rem;">
  <img src="{{ '/images/CIFApredood.png' | relative_url }}" alt="High noise" width="45%">
</div>
<p style="font-size: 0.9rem; color: #666; margin-top: 0.2rem;">
  Figure: Average prediction set size vs \(\alpha\), CIFAR-100 (out-of-distribution).
</p>



# 03 — Training Concepts

---

## 1. Loss Function

Measures **how wrong the model is**. Training = minimizing this number.

For language models the loss is **cross-entropy**:

```
Loss = -log(P(correct next token))
```

- Model predicts 0.8 probability for correct token → loss = -log(0.8) = 0.22  ✓
- Model predicts 0.01 probability for correct token → loss = -log(0.01) = 4.6  ✗

Averaged across every token in every batch.

### Perplexity:
Human-readable version of loss:
```
Perplexity = e^loss
```
Means: "on average the model is as confused as choosing between N options"
- Random guess over 50k vocab → perplexity = 50,000
- GPT-2 small → perplexity ≈ 29
- GPT-4 class → perplexity ≈ 5–8

---

## 2. Optimizer

Takes gradients (direction of steepest loss increase) and decides how to update weights.

### SGD (Stochastic Gradient Descent) — baseline:
```
weight = weight - learning_rate × gradient
```
Simple. Works. But slow and sensitive to learning rate.

### Adam (Kingma & Ba, 2014) — dominant:
Tracks two things per weight:

```
m = β1 * m + (1 - β1) * gradient        ← moving average of gradient (momentum)
v = β2 * v + (1 - β2) * gradient²       ← moving average of gradient² (variance)

weight = weight - lr * m / (sqrt(v) + ε)
```

- **m** smooths out noisy gradients — don't overreact to one bad batch
- **v** scales the step — weights that wiggle a lot get smaller updates
- Net effect: each weight gets its own adaptive learning rate

### AdamW:
Adam + weight decay applied directly to weights (not via gradients).
Standard for transformers today.

```
weight = weight * (1 - lr * λ) - lr * adam_update
```

λ = weight decay coefficient, typically 0.1.

---

## 3. Initialization Scale

Before training, weights are random. **How random matters a lot.**

If weights start too large → activations explode → gradients explode → NaN loss.
If weights start too small → activations vanish → gradients vanish → no learning.

### Xavier / Glorot init:
```
W ~ Uniform(-√(6/(fan_in + fan_out)), +√(6/(fan_in + fan_out)))
```
Designed to keep variance stable through layers. Good for tanh/sigmoid.

### Kaiming / He init:
```
W ~ Normal(0, √(2 / fan_in))
```
Designed for ReLU. Accounts for the fact that ReLU kills half the values.

### GPT-style:
```
W ~ Normal(0, 0.02)
```
Small fixed std. Residual layers scaled down by 1/√(2 * n_layers) to prevent
variance from accumulating as you stack layers.

> Rule of thumb: at init, the std of activations should be ~1.0 throughout the network.
> If it drifts up or down layer by layer, training will be unstable.

---

## 4. Learning Rate Schedule

The learning rate (lr) controls step size. Fixed lr rarely works best.

### Warmup:
Start with tiny lr, ramp up over first ~1000 steps.
Why: at init weights are random → gradients are noisy → big steps = chaos.
Warmup lets the model stabilize before taking large steps.

### Cosine decay:
After warmup, decay lr following a cosine curve down to near zero:
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

```
lr
|  /\
| /  \
|/    \__________
+---------------→ steps
warmup  cosine decay
```

### Linear decay:
Simpler alternative. Decay linearly from lr_max to lr_min.

### Typical GPT training schedule:
```
Steps 0 → 2000:     linear warmup  0 → 3e-4
Steps 2000 → end:   cosine decay   3e-4 → 3e-5
```

---

## 5. Regularization

Prevents the model from memorizing training data instead of learning patterns (overfitting).

### Weight decay:
Penalizes large weights. Forces the model to find simpler solutions.
```
total_loss = cross_entropy_loss + λ * sum(w²)
```
Applied via AdamW. λ typically 0.1.

### Dropout:
During training, randomly zero out a fraction of activations.
```
p = 0.1 means 10% of neurons output 0 each forward pass
```
Forces the network to not rely on any single neuron. Disabled at inference.
Less used in modern large models — weight decay + data scale does the job.

### Gradient clipping:
Not quite regularization but prevents explosions:
```
if ||gradient|| > threshold:
    gradient = gradient * threshold / ||gradient||
```
Caps the gradient norm. Typical threshold = 1.0.

---

## 6. Batch Size

How many sequences you process before updating weights.

```
Small batch (32):     noisy gradients → model generalizes better, but slow
Large batch (1024+):  stable gradients → fast training, but can overfit
```

### Gradient accumulation:
If GPU can't fit large batch → run N small batches, accumulate gradients, then update once.
Simulates a larger batch without needing more GPU memory.

### Critical batch size:
There's a point beyond which making the batch bigger stops helping.
Beyond that point you're wasting compute — doing fewer updates per token seen.

### Rule of thumb for LLMs:
Batch size is often measured in **tokens not sequences**:
```
GPT-3 training batch = 3.2M tokens per step
```
Start small, scale batch size up as model grows.

---

## 7. Mixture of Experts (MoE)

Normal transformer: every token passes through **every** MLP neuron.
MoE: each token is routed to only a **few** expert MLPs out of many.

### Architecture:

```
Normal MLP:
  token → [single MLP] → output

MoE MLP:
  token → [Router] → picks top-K experts out of N
        → [Expert_3, Expert_7] → weighted sum → output
```

Router is a small learned network that looks at the token and picks which experts to use.

### Why it's powerful:

```
Normal GPT-3:    175B params — all active per token
GPT-4 (rumored): ~1.8T params — 8 experts, 2 active → ~220B active per token
```

You get a huge model (total params) at the cost of a much smaller model (active params per token).
**More capacity, same compute.**

### Challenges:

| Problem | What happens | Fix |
|---------|-------------|-----|
| Load imbalance | All tokens route to same expert | Auxiliary load-balancing loss |
| Expert collapse | Some experts never get used | Same |
| Communication cost | Experts may be on different GPUs | Careful placement |

### Real models using MoE:
- Mixtral 8×7B (Mistral) — 8 experts, 2 active, open source
- GPT-4 (rumored)
- Gemini 1.5
- Switch Transformer (Google, 2021) — first large MoE paper

### MoE vs dense:
```
Same compute budget:
  Dense model:  all params active, smaller total params
  MoE model:    few params active, larger total params → better quality
```

MoE wins on quality per FLOP. Loses on serving complexity.

---

## Summary

```
Loss function      — measures how wrong (cross-entropy → perplexity)
Optimizer          — AdamW: adaptive lr per weight + weight decay
Initialization     — keep activation std ≈ 1.0 throughout the network
LR schedule        — warmup then cosine decay
Regularization     — weight decay + gradient clipping (dropout optional)
Batch size         — measured in tokens; gradient accumulation for large batches
MoE                — route each token to top-K of N experts; more params, same compute
```

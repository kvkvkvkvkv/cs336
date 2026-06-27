# ML Fundamentals — FLOPs, MLP, Attention

## The Evergreen Stack

Models change every 6 months. These four do not:

```
Attention       — how models focus and relate information
Kernels         — low-level GPU math ops (CUDA, cuDNN, Flash Attention)
Optimization    — how models learn (SGD, Adam, learning rate schedules)
GPUs            — the hardware everything runs on
```

### Why each is durable

| Fundamental | Why it won't go away |
|-------------|----------------------|
| **Attention** | Every serious architecture since 2017 uses it; MoE, SSMs etc. still build on or compete with it |
| **Kernels** | Raw compute is always the bottleneck; writing fast GPU kernels = direct speed gains regardless of model |
| **Optimization** | You always need to move loss downhill; Adam variants dominate but the math is the same |
| **GPUs** | Training and inference run here; understanding memory hierarchy, bandwidth, and parallelism is timeless |

> Learn the model of the week and you're current for a month. Learn these four and you can read any paper from 2017–2035.

---


## FLOPs (Floating Point Operations)

A measure of compute. One FLOP = one addition, multiplication, or similar math operation on a decimal number.

Used to quantify:
- How much compute a model needs to run (inference FLOPs)
- How much was used to train it (training FLOPs)

### Where do FLOPs live?

FLOPs live **inside each neuron's math**.

A single neuron does this:
```
output = activation(w1*x1 + w2*x2 + w3*x3 + bias)
```

- Each `w*x` multiplication = **1 FLOP**
- Each `+` addition = **1 FLOP**

So one neuron with 3 inputs = ~6 FLOPs.

### Scale that up to MLP:

```
Input layer:   784 neurons
Hidden layer:  512 neurons
Output layer:  10 neurons
```

Input → Hidden: 784 × 512 = **~800K FLOPs** (for one forward pass through one layer)

### Scale to GPT-4:

- Billions of neurons
- Run millions of times during training
- Each run = trillions of FLOPs

> FLOPs count how many `w*x` multiplications and additions happen. More neurons + more layers + more data = more FLOPs.

---

## MLP (Multi-Layer Perceptron)

The "feed-forward" part of a neural network. Stacked layers of:

```
input → [Linear → Activation] → [Linear → Activation] → ... → output
```

In transformers, every layer has two sub-components:
1. **Attention** — lets tokens talk to each other
2. **MLP** — processes each token independently through a small neural net

The MLP is where most of the model's "memory" (factual knowledge) is thought to live.

---

## Attention

Attention sits **between** the input and the MLP — **separate from neurons**.

### Transformer layer structure:

```
Input tokens
     ↓
[ Attention ]      ← tokens talk to each other
     ↓
[    MLP    ]      ← each token processed independently (neurons live here)
     ↓
Output
```

### What attention does:

For the sentence: `"The cat sat on the mat"`

Before MLP processes "sat", attention asks:
> "which other words should I look at to understand this word?"

It looks at "cat" and "mat" and blends their information into "sat". That blending = attention. No neurons involved.

The math:
```
Attention(Q, K, V) = softmax(Q × K^T) × V
```

Just matrix multiplications — no activation function, no weights-per-neuron.

### Attention vs MLP:

| Part | Job | Analogy |
|------|-----|---------|
| Attention | Decide what to focus on | Eyes scanning the room |
| MLP (neurons) | Process that focused info | Brain thinking about it |

Both attention and MLP cost FLOPs — just different math.

---

## SwiGLU (Swish-Gated Linear Unit)

An **activation function** used inside the MLP layer of modern transformers (LLaMA, PaLM, Gemini).

### The MLP neuron recap:

```
output = activation(w * x)
```

The `activation` adds non-linearity. Options:
- **ReLU** — `max(0, x)` — kills negative values dead
- **GELU** — smoother ReLU
- **SwiGLU** — newer, works better in practice

### What SwiGLU does differently:

Two paths instead of one, then multiply them:

```
SwiGLU(x) = Swish(W1 * x)  ×  (W2 * x)
```

- **Left path** — `Swish(W1 * x)` — acts as a gate (values between 0 and 1)
- **Right path** — `W2 * x` — the actual information
- **Multiply** — gate controls how much info passes through

The gate learns to **open or close** for each piece of information.

### Analogy:

> Right path = water flowing through a pipe
> Left path = a valve controlling the flow
> SwiGLU = valve × water

### Where it's used:

| Activation | Used in |
|------------|---------|
| ReLU | Old transformers, ResNets |
| GELU | GPT-2, BERT |
| SwiGLU | LLaMA, PaLM, Gemini |

Empirically SwiGLU trains to **lower loss** with the same compute. The theory of why is still fuzzy.

> SwiGLU is a smarter activation inside MLP neurons — uses a learned gate to control which information flows through, instead of a fixed math function.

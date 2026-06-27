# 02 — Transformer Components

## The transformer block (one layer):

```
Input (tokens as vectors)
   ↓
Positional Encoding    — inject position info
   ↓
LayerNorm              — normalize
   ↓
Attention              — tokens talk to each other
   ↓  (+ residual)
LayerNorm              — normalize again
   ↓
MLP                    — each token processed independently
   ↓  (+ residual)
Output
```

Repeat this block N times (GPT-2 small = 12 layers, GPT-3 = 96 layers).

---

## 1. Activation

A math function applied after a linear layer to add **non-linearity**.

Without activation, stacking linear layers = still just one linear layer. Useless.

```
Linear:      output = W * x        (just a line)
With ReLU:   output = max(0, W*x)  (can now bend)
```

### Common activations:

| Name | Formula | Used in |
|------|---------|---------|
| ReLU | max(0, x) | Old networks |
| GELU | x * Φ(x) (smooth ReLU) | GPT-2, BERT |
| SwiGLU | Swish(W1x) × W2x | LLaMA, Gemini |

> See ml-fundamentals.md for SwiGLU deep dive.

---

## 2. Positional Encoding

Transformers process all tokens **in parallel** — they have no built-in sense of order.

"cat sat" and "sat cat" would look identical without positional encoding.

**Fix:** add a position vector to each token embedding before the first layer.

```
token_vector = embedding("cat") + position_vector(position=2)
```

### Types:

**Sinusoidal (original Transformer, 2017):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Fixed math formula. No learned params. Works but doesn't generalize well to longer sequences.

**Learned absolute (GPT-2):**
Each position 1..N gets a learned vector. Simple but can't generalize beyond training length.

**RoPE — Rotary Position Embedding (LLaMA, GPT-NeoX):**
Instead of adding position to the token, it **rotates** the query and key vectors by an angle proportional to position. Relative distances are preserved.
- Generalizes to longer sequences than trained on
- Now the dominant method

**ALiBi (attention with linear biases):**
Subtracts a penalty from attention scores based on distance between tokens. No vectors added, just a bias.

---

## 3. Normalization

Training deep networks is unstable — activations can explode or vanish.
Normalization keeps values in a stable range at every layer.

### LayerNorm (BERT, GPT-2):
```
x_norm = (x - mean(x)) / std(x)   ← normalize across features
output = gamma * x_norm + beta     ← learned scale and shift
```
Applied **before** attention and MLP (Pre-LN, modern default).

### RMSNorm (LLaMA, T5):
```
x_norm = x / rms(x)    ← rms = sqrt(mean(x²))
output = gamma * x_norm
```
Drops the mean subtraction and beta. Faster, works just as well in practice.
Now preferred over LayerNorm.

---

## 4. Attention

Tokens ask: *"which other tokens should I look at?"*

### The math:

```
Q = X * W_Q    (Query — what am I looking for?)
K = X * W_K    (Key   — what do I offer?)
V = X * W_V    (Value — what info do I carry?)

Attention(Q,K,V) = softmax(Q * K^T / sqrt(d_k)) * V
```

- `Q * K^T` → dot product = similarity score between every pair of tokens
- `/ sqrt(d_k)` → scale down to stop softmax from saturating
- `softmax(...)` → turn scores into probabilities (attention weights)
- `* V` → weighted sum of value vectors

### Multi-head attention:
Run attention H times in parallel with different W_Q, W_K, W_V each time.
Each head learns to attend to different kinds of relationships.

```
head_i = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)
MultiHead = concat(head_1, ..., head_H) * W_O
```

### Complexity:
Attention is **O(n²)** in sequence length — every token attends to every other.
Fine for 2K tokens. Painful at 100K+.

---

## 5. Recurrence / State Space Models / Linear Attention

All three are responses to attention's O(n²) problem.

### Recurrence (LSTM, GRU — pre-transformer):
```
h_t = f(h_{t-1}, x_t)
```
Process one token at a time. Hidden state `h` carries memory forward.
- Pro: O(n) — fast
- Con: hard to parallelize during training; memory bottleneck (can't look back far)

### State Space Models — SSM (Mamba, S4):
Mathematical evolution of RNNs. Uses a state matrix to compress the past.
```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t
```
A, B, C are learned matrices. Can be computed as a convolution during training (parallel) and as recurrence during inference (fast).
- Pro: O(n) inference, parallelizable training
- Con: still a fixed-size state — theoretically can forget

### Linear Attention:
Rewrites the attention formula to avoid materializing the full n×n matrix:
```
Normal:   softmax(Q K^T) V    ← O(n²) because Q K^T is n×n
Linear:   φ(Q) (φ(K)^T V)    ← O(n) because φ(K)^T V is d×d
```
Where φ is a kernel function that approximates softmax.
- Pro: O(n) — scales to long sequences
- Con: approximation loses some expressiveness vs full attention

### Status today:
```
Short sequences (< 8K):    Attention wins
Long sequences (> 32K):    SSMs / linear attention competitive
Hybrid (Jamba, etc.):      Mix of attention + SSM layers
```

---

## 6. MLP and Shape

### MLP inside a transformer:

```
x  →  Linear(d_model → d_ff)  →  Activation  →  Linear(d_ff → d_model)  →  output
```

`d_ff` is typically **4 × d_model** (e.g. 768 → 3072 → 768 in GPT-2 small).

### Shape — the tensor dimensions:

Every tensor flowing through a transformer has this shape:

```
[B, T, D]

B = batch size        (how many sequences processed in parallel)
T = sequence length   (number of tokens)
D = d_model           (embedding dimension, e.g. 768)
```

### How shape changes through the model:

```
Input tokens:              [B, T]           ← integers
After embedding:           [B, T, D]        ← floats
After attention:           [B, T, D]        ← same shape (attention mixes T, keeps D)
After MLP:                 [B, T, D]        ← same shape (MLP transforms D, independent per T)
After final linear:        [B, T, vocab]    ← project to vocabulary size
After softmax:             [B, T, vocab]    ← probabilities
```

Attention mixes across **T** (tokens talk to each other).
MLP transforms across **D** (each token's features transformed independently).
Shape in = shape out for both. That's why residual connections work cleanly.

### GPT-2 small numbers:
```
d_model  = 768
d_ff     = 3072   (4 × 768)
n_heads  = 12
d_head   = 64     (768 / 12)
n_layers = 12
vocab    = 50,257
```

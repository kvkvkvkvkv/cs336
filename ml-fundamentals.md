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

---

## Optimistic Locking vs CAS

**CAS (Compare-And-Swap)** is a *primitive*: one atomic CPU instruction (`LOCK CMPXCHG` on x86) that says "if the value at this address is still X, set it to Y; otherwise fail."

**Optimistic locking** is a *strategy*: don't lock while you work; at write time, check nobody changed the data since you read it. If someone did, retry or abort.

> Optimistic locking is the idea. CAS is one way to implement it, at the memory level.

### Same pattern, different layers:

| Layer | Mechanism | Fails how |
|-------|-----------|-----------|
| CPU / memory | `AtomicLong.compareAndSet(expected, new)` | returns `false` |
| Database | `UPDATE ... SET version = v+1 WHERE id = ? AND version = v` | 0 rows updated → `OptimisticLockException` (JPA `@Version`) |
| Redis | `WATCH key` + `MULTI/EXEC` | `EXEC` returns nil |
| DynamoDB / etcd | conditional write / txn on revision | condition check fails |

The DB version check *is* a CAS — on a row instead of a memory word.

### Key differences:

| | CAS | Optimistic locking (DB) |
|--|-----|-----|
| Scope | one memory word | a whole row / entity (many fields) |
| Cost of a failed attempt | nanoseconds, spin & retry | a DB round trip, maybe redo business logic |
| ABA problem | yes — value goes A→B→A, CAS wrongly succeeds; fix with `AtomicStampedReference` | no — version only increments, never repeats |

### When it breaks down:
Both assume **low contention**. Under heavy contention, most attempts fail and retry → a retry storm that wastes more than a lock would.

- Low contention → optimistic
- High contention on one row → pessimistic (`SELECT ... FOR UPDATE`) or **single writer**

### Trading example (CoinDCX):
- Normal user's wallet debit: `UPDATE wallet SET balance = balance - 100, version = version + 1 WHERE user_id = ? AND version = ? AND balance >= 100` — rare conflicts, optimistic is fine.
- Market maker's account, thousands of updates/sec: optimistic retries explode. Route all of that account's updates to one thread (e.g. Kafka partition keyed by account), so there's no contention at all. That's why matching engines are single-writer.

> CAS = hardware "check-then-set" on one value. Optimistic locking = the same check-then-set applied to data at any layer, usually with a version number.

---

## BPE (Byte Pair Encoding) — the tokenizer

**What it does:** turns text into a list of integer IDs (tokens) that the model can consume. It decides *how to chop text into pieces*.

### The problem it solves:
| Approach | Problem |
|---|---|
| One token per **word** | Vocab explodes (millions), and unseen words ("CoinDCX", typos) → `<UNK>` |
| One token per **byte/char** | Never unknown, but sequences get very long → attention is O(n²), so expensive |
| **BPE** (subwords) | Middle ground: common words = 1 token, rare words = a few pieces, nothing is ever unknown |

### Training (learning the merges):
1. Start with the 256 possible bytes as the vocab. Every string can be written in bytes, so nothing is ever unknown.
2. Count every adjacent pair across the corpus.
3. Take the most frequent pair, e.g. (`t`, `h`), and turn it into a new token `th`. Write down the merge rule.
4. Replace that pair everywhere, recount, repeat.
5. Stop once vocab hits your target size (GPT-2: ~50k; CS336 TinyStories: 10k).

Output = **vocab** (id → bytes) + **ordered merge list**.

Tiny example, corpus `"low lower lowest"`:
- `l o` most frequent → merge `lo`
- `lo w` → merge `low`
- now `low` is a single token; `lower` = `low` + `e` + `r`, and so on.

### Encoding (using it on new text):
Turn the text into bytes, then apply the learned merges **in the order they were learned**. Then map each piece to its ID.

### Pre-tokenization (important detail):
Before counting, split the text with a regex into word-like chunks (GPT-2 pattern) and **never merge across chunk boundaries**. Otherwise you'd get junk tokens like `dog.` vs `dog!` vs `dog?`. Leading spaces stick to words: `" the"` is a token.

### Why it matters:
- Context length, cost, and pricing are all counted in **tokens**, not words. Roughly 1 token ≈ 4 English chars.
- Weird LLM failures come from tokenization: counting letters in "strawberry", arithmetic (numbers get split strangely), and non-English text costing 2–5× more tokens.

### Systems analogy:
Like building a compression dictionary (LZW): frequent patterns get short codes. BPE was literally a compression algorithm first (1994).

> BPE = start from bytes, repeatedly glue together the most frequent adjacent pair. You end up with a vocab where common stuff is one token and rare stuff is a few pieces, and nothing is ever unknown.

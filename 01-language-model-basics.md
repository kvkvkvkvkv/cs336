# 01 — Training a Basic Language Model

## The Big Picture

```
Raw Text
   ↓
Tokenization       — text → numbers the model can read
   ↓
Model Architecture — numbers → predictions (what comes next?)
   ↓
Training           — measure how wrong, adjust weights, repeat
```

---

## Step 1 — Tokenization

The model cannot read text. It reads integers. Tokenization converts text → integers.

### How it works:

```
"Hello world"  →  [15496, 995]
```

Each integer = a **token** (a chunk of text, not always a full word)

### Types:

| Type | Example | Used in |
|------|---------|---------|
| Word-level | "hello" = 1 token | old NLP |
| Character-level | "h","e","l","l","o" = 5 tokens | simple models |
| BPE (Byte Pair Encoding) | "hel","lo" = 2 tokens | GPT family |
| WordPiece | "hell","##o" = 2 tokens | BERT family |

### BPE (what GPT uses) in plain English:

1. Start with every character as its own token
2. Find the most common pair of tokens → merge into one
3. Repeat until vocabulary size is hit (GPT-2 = 50,257 tokens)

```
"low lower lowest"
→ start: l o w | l o w e r | l o w e s t
→ merge "lo": lo w | lo w e r | lo w e s t
→ merge "low": low | low e r | low e s t
→ merge "er": low | lower | low e s t
...
```

**Vocabulary size tradeoff:**
- Too small → common words split into many tokens → slow, loses meaning
- Too large → rare tokens never seen in training → wasted capacity

---

## Step 2 — Model Architecture

A language model takes tokens in and predicts **the next token**.

```
Input:  ["The", "cat", "sat"]
Output: probability over all 50,257 tokens for what comes next
        → "on" gets highest probability
```

### Inside the model:

```
Tokens (integers)
   ↓
Embedding Layer        — integer → dense vector (e.g. 768 numbers)
   ↓
Transformer Blocks ×N  — each block = Attention + MLP
   ↓
Linear + Softmax       — vector → probability over vocabulary
```

### Embedding layer:

Token `15496` → look up row 15496 in a learned table → get a vector of 768 numbers.

That vector = the model's internal representation of that token.

### Transformer block (repeated N times):

```
x → [LayerNorm → Attention → +x] → [LayerNorm → MLP → +x] → x
```

The `+x` is a **residual connection** — output gets added back to input.
Prevents information from being lost as it passes through layers.

### Final layer:

```
Vector (768 numbers) → Linear → 50,257 numbers → Softmax → probabilities
```

Softmax turns raw numbers into probabilities that sum to 1.

---

## Step 3 — Training

Goal: make the model assign **high probability to the correct next token**.

### The training loop:

```
1. Take a chunk of text:        "The cat sat on the mat"
2. Shift by 1 to get targets:   "cat sat on the mat <end>"
3. Forward pass → model outputs probabilities for each position
4. Compute loss (how wrong is it?)
5. Backprop → compute gradients
6. Optimizer step → update weights
7. Repeat on next batch
```

### Loss function — Cross Entropy:

```
Loss = -log(probability assigned to correct token)
```

- Model says "on" has 0.8 probability → loss = -log(0.8) = 0.22 ✓ low loss
- Model says "on" has 0.01 probability → loss = -log(0.01) = 4.6 ✗ high loss

**Perplexity** = e^loss — a human-readable version. Lower = better.
- Random guessing on 50k vocab → perplexity ≈ 50,000
- GPT-2 small → perplexity ≈ 29 on standard benchmarks

### Backpropagation:

Chain rule applied backwards through the network.
Computes: "if I nudge this weight slightly, how much does the loss change?"
That nudge direction = gradient.

### Optimizer — Adam:

```
weight = weight - learning_rate × gradient
```

But Adam makes this smarter:
- Tracks a **moving average of gradients** (momentum — don't react to noise)
- Tracks a **moving average of gradient²** (scale step by how noisy each weight is)

Result: weights that rarely move get bigger updates; weights that oscillate get smaller updates.

---

## Summary

```
Text → Tokens (BPE)
         ↓
Tokens → Embeddings (lookup table)
         ↓
Embeddings → Transformer blocks (Attention + MLP) × N
         ↓
Output → probabilities over vocab
         ↓
Loss = -log(P(correct token))
         ↓
Backprop → gradients
         ↓
Adam → update weights
         ↓
Repeat millions of times
```

The model has learned when loss stops dropping on held-out text (validation loss).

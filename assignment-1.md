# Assignment 1 — Build a Language Model from Scratch

## Tasks

- [ ] BPE Tokenizer
- [ ] Transformer architecture
- [ ] Cross-entropy loss
- [ ] AdamW optimizer
- [ ] Training loop
- [ ] Resource accounting (FLOPs, memory, time)
- [ ] Train on TinyStories
- [ ] Train on OpenWebText

## Datasets

**TinyStories** — small synthetic dataset of children's stories. Fast to train on. Good for sanity-checking the pipeline works before scaling up.

**OpenWebText** — open reproduction of WebText (what GPT-2 was trained on). Much larger. Real benchmark.

## Order to build

```
1. BPE tokenizer      → text → integers
2. Transformer        → integers → logits
3. Cross-entropy loss → logits → scalar loss
4. AdamW optimizer    → loss → weight updates
5. Training loop      → tie it all together
6. Resource accounting → measure FLOPs/memory/throughput
7. Train TinyStories  → sanity check (should see loss drop fast)
8. Train OpenWebText  → real run
```

## Notes from session

- BPE: start with bytes, merge most frequent pairs, repeat until vocab size hit
- Transformer shape: [B, T, D] flows unchanged through attention + MLP
- Cross-entropy: -log(P(correct token)), averaged over all tokens in batch
- AdamW: momentum + adaptive lr per weight + weight decay directly on weights
- Resource accounting: track tokens/sec, MFU (model FLOP utilization), GPU memory

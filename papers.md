# Reading List — Foundational ML Papers

## To Read

| Paper | Authors | Year | Topic |
|-------|---------|------|-------|
| Measuring the Algorithmic Efficiency of Neural Networks | Hernandez & Brown | 2020 | ImageNet efficiency / algorithmic progress |
| Long Short-Term Memory | Hochreiter & Schmidhuber | 1997 | LSTM |
| A Neural Probabilistic Language Model | Bengio et al. | 2003 | First neural language model |
| Sequence to Sequence Learning with Neural Networks | Sutskever et al. | 2014 | Seq2Seq |
| Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | Adam optimizer |
| Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau et al. | 2014 | Attention mechanism |
| Attention Is All You Need | Vaswani et al. | 2017 | Transformer architecture |
| Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Shazeer et al. | 2017 | Mixture of experts |
| GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | Huang et al. | 2019 | Model parallelism |

## Early Foundation Models

| Paper | Authors | Year | Key idea |
|-------|---------|------|----------|
| Deep contextualized word representations (ELMo) | Peters et al. | 2018 | Bidirectional LSTMs to produce context-aware embeddings — same word gets different vector depending on sentence |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin et al. (Google) | 2018 | Transformer encoder trained with masked language modeling; first model to dominate NLP benchmarks |
| Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) | Raffel et al. (Google) | 2019 | 11B param model; frames every NLP task as text-in → text-out; trained on Colossal Clean Crawled Corpus (C4) |

### Quick mental model

```
ELMo (2018)        — LSTM-based, contextual embeddings, pre-transformer
    ↓
BERT (2018)        — Transformer encoder, bidirectional, masked LM pretraining
    ↓
T5 (2019, 11B)     — Encoder-decoder, unified text-to-text format, massive scale
```

- **ELMo** → proved context matters (same word, different meaning = different vector)
- **BERT** → proved transformers + pretraining beats everything else
- **T5** → proved one model + one format can do all NLP tasks at scale

## Read

_(move papers here once done)_

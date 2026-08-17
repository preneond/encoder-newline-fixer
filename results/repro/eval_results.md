# Evaluation results

Data: `data/docs/test.jsonl` — 120 docs, seed 13.

| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim | Exact | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|---|
| encoder | 0.9811 | 0.8136 | 0.7690 | 0.2294 | 0.3002 | 0.9944 | 0.008 | 2,931 | 648.24 |

\* Macro-F1 over {JOIN, NEWLINE, PARA} (SPACE excluded as the trivial majority class).

## Qualitative example (README)

Input:

```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
```

### encoder

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

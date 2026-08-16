# Evaluation results

Data: `data/docs/test.jsonl` — 120 docs, seed 13.

| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim | Exact | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|---|
| distilled-a09 | 0.9653 | 0.6339 | 0.6265 | 0.3219 | 0.4203 | 0.9894 | 0.000 | 2,706 | 702.18 |

\* Macro-F1 over {JOIN, NEWLINE, PARA} (SPACE excluded as the trivial majority class).

## Qualitative example (README)

Input:

```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
```

### distilled-a09

```
3.2.3 Applications of Attention in our Model The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the que ries come from the previous decoder layer.[...]
```

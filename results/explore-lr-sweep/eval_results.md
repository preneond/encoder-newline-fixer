# Evaluation results

Data: `data/docs/test.jsonl` — 120 docs, seed 13.

| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim | Exact | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|---|
| lr1e-4 | 0.9812 | 0.8148 | 0.7687 | 0.2396 | 0.3109 | 0.9941 | 0.000 | 9,221 | 206.06 |
| lr2e-4 | 0.9804 | 0.8093 | 0.7600 | 0.2504 | 0.3203 | 0.9939 | 0.000 | 18,192 | 104.45 |
| lr2e-5 | 0.9718 | 0.7291 | 0.6955 | 0.2992 | 0.4096 | 0.9913 | 0.000 | 18,096 | 105.00 |
| distilled | 0.9682 | 0.6749 | 0.6572 | 0.3227 | 0.4173 | 0.9897 | 0.000 | 2,740 | 693.54 |

\* Macro-F1 over {JOIN, NEWLINE, PARA} (SPACE excluded as the trivial majority class).

## Qualitative example (README)

Input:

```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
```

### lr1e-4

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

### lr2e-4

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

### lr2e-5

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

•

In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

### distilled

```
3.2.3 Applications of Attention in our Model The Transformer uses multi-head attention in three different ways:

•

In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

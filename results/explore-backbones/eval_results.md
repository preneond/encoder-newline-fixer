# Evaluation results

Data: `data/docs/test.jsonl` — 120 docs, seed 13.

| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim | Exact | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|---|
| encoder | 0.9811 | 0.8136 | 0.7690 | 0.2294 | 0.3002 | 0.9944 | 0.008 | 2,044 | 929.79 |
| electra-small | 0.9677 | 0.7015 | 0.6654 | 0.2823 | 0.3956 | 0.9902 | 0.000 | 4,727 | 401.94 |
| distilbert-cased | 0.9741 | 0.7627 | 0.7102 | 0.2775 | 0.3658 | 0.9925 | 0.000 | 1,974 | 962.56 |
| roberta-base | 0.9824 | 0.8254 | 0.7830 | 0.2094 | 0.2743 | 0.9947 | 0.000 | 1,039 | 1828.91 |

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

### electra-small

```
3.2.3 Applications of Attention

in our Model

The Transformer uses multi-head attention in three different ways:

•

In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

### distilbert-cased

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

•

In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

### roberta-base

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

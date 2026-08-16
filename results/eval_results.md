# Evaluation results

Data: `data/docs/test.jsonl` — 120 docs, seed 13.

| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim | Exact | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|---|
| majority | 0.9572 | 0.0000 | 0.0000 | 0.3409 | 0.3409 | 0.9864 | 0.000 | 1,398,543,349 | 0.00 |
| rules | 0.9544 | 0.0342 | 0.0399 | 0.3482 | 0.3519 | 0.9858 | 0.000 | 10,539,407 | 0.18 |
| encoder | 0.9787 | 0.7891 | 0.7484 | 0.2493 | 0.3283 | 0.9934 | 0.000 | 13,802 | 137.67 |
| scratch | 0.9590 | 0.6319 | 0.6172 | 0.3648 | 0.4845 | 0.9874 | 0.000 | 2,172 | 874.88 |

\* Macro-F1 over {JOIN, NEWLINE, PARA} (SPACE excluded as the trivial majority class).

## Qualitative example (README)

Input:

```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
```

### majority

```
3.2.3 Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the que ries come from the previous decoder layer.[...]
```

### rules

```
3.2.3 Applications of Attention in our Model The Transformer uses multi-head attention in three different ways:
• In "encoder-decoder attention" layers, the que ries come from the previous decoder layer.[...]
```

### encoder

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

### scratch

```
3.2.3

Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:

•

In "encoder-decoder attention" layers, the queries come from the previous decoder layer.[...]
```

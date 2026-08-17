# Exploration results

Data: `data/docs/test.jsonl` — 120 docs, seed 13.

| Model | Gap acc | Macro-F1* | Break-F1 | Pk | WinDiff | EditSim | Exact | Words/s | ms/doc |
|---|---|---|---|---|---|---|---|---|---|
| majority | 0.9572 | 0.0000 | 0.0000 | 0.3409 | 0.3409 | 0.9864 | 0.000 | 878,649,826 | 0.00 |
| rules | 0.9544 | 0.0342 | 0.0399 | 0.3482 | 0.3519 | 0.9858 | 0.000 | 10,161,222 | 0.19 |
| encoder | 0.9812 | 0.8148 | 0.7687 | 0.2396 | 0.3109 | 0.9941 | 0.000 | 10,288 | 184.70 |
| scratch | 0.9590 | 0.6319 | 0.6172 | 0.3648 | 0.4845 | 0.9874 | 0.000 | 2,801 | 678.31 |
| lr1e-4 | 0.9812 | 0.8148 | 0.7687 | 0.2396 | 0.3109 | 0.9941 | 0.000 | 9,221 | 206.06 |
| lr2e-4 | 0.9804 | 0.8093 | 0.7600 | 0.2504 | 0.3203 | 0.9939 | 0.000 | 18,192 | 104.45 |
| lr2e-5 | 0.9718 | 0.7291 | 0.6955 | 0.2992 | 0.4096 | 0.9913 | 0.000 | 18,096 | 105.00 |
| distilled | 0.9682 | 0.6749 | 0.6572 | 0.3227 | 0.4173 | 0.9897 | 0.000 | 2,740 | 693.54 |
| distilled-a09 | 0.9653 | 0.6339 | 0.6265 | 0.3219 | 0.4203 | 0.9894 | 0.000 | 2,706 | 702.18 |
| electra-small | 0.9677 | 0.7015 | 0.6654 | 0.2823 | 0.3956 | 0.9902 | 0.000 | 4,727 | 401.94 |
| distilbert-cased | 0.9741 | 0.7627 | 0.7102 | 0.2775 | 0.3658 | 0.9925 | 0.000 | 1,974 | 962.56 |
| roberta-base | 0.9824 | 0.8254 | 0.7830 | 0.2094 | 0.2743 | 0.9947 | 0.000 | 1,039 | 1828.91 |

Backbone-sweep rows (electra-small, distilbert-cased, roberta-base) were measured
on an M4 MacBook Air; earlier rows on an M4 Max — words/s is not comparable across
the two groups. roberta-base was trained on CPU (torch 2.13.0 MPS NaN bug; see
report.md "Backbone sweep").

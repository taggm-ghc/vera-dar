# Evaluation and findings

[← Project overview](../README.md)

VERA-DAR's principal evaluation question is whether evidence-grounded context construction results in materially better research answers than an otherwise comparable model answering directly. The available experiments are informative but do not establish that claim.

## Historical TRACE comparison

A 2026-10-04 experiment used 30 frozen, agent-authored questions (20 development, 10 held-out). Two variants used `gpt-4.1-nano`: a question-only baseline and a grounded variant supplied with up to five numbered arXiv abstracts. Both were scored using the same frozen deterministic checks.

| Measure | Question-only | Grounded variant | Interpretation |
|---|---:|---:|---|
| Citation present, 22 applicable traces | 1/22 (5%) | 10/22 (45%) | Difference reported with paired exact McNemar p = 0.0039 |
| No phantom evidence, 22 applicable traces | 4/22 (18%) | 17/22 (77%) | Difference reported with paired exact McNemar p = 0.0002 |
| All checks pass, development set | 2/20 (10%) | 3/20 (15%) | No distinguishable improvement |
| All checks pass, held-out set | 2/10 (20%) | 2/10 (20%) | No improvement |

The p-values are unadjusted and come from a small, single-run sample. The statistically significant results concern those specific citation-related checks, **not overall answer superiority**. A small sample and correlated design and labelling decisions limit generalization.

## What the failures suggest

The analysis of that experiment found that the grounded variant reduced some obvious unsupported-evidence patterns but frequently retrieved irrelevant sources. In 13 of 19 grounded development traces, no more than one of the five retrieved results was on-topic. Consequently, the model sometimes declined rather than inventing an apparently supported answer.

The analysis also reports that four deterministic checks over-flagged some passing outputs (true-negative rates of approximately 0.50–0.75) and that one word-overlap heuristic failed to detect partial support. These checks cannot be treated as infallible quality judges.

## Later, non-comparable measurement

A separate evaluation of a later abstract-grounded `/ask` configuration (since superseded by the default reading-list behaviour) reported 19 of 30 questions answered with checked citations, zero unresolvable citations among 57, and 9 of 105 checked claims classified as unsupported (8.6%, with reported Wilson 95% interval 4.6%–15.5%) before removal from the displayed answer.

**This is not a before-and-after estimate.** That later run did not have a direct-model baseline evaluated under the same claim-check procedure. It therefore does not prove a gain attributable to the retrieval and checking changes.

## Remaining research questions

- How much does improved retrieval alter answer quality rather than only citation presence?
- Can provenance be tied to inspected full-document passages instead of abstracts alone?
- What is the independent error rate of the claim checker?
- Do benefits persist on genuinely external questions, independent graders and strong current baselines?
- What are the latency, cost and reliability trade-offs of added evidence processing?

Internal trace files, frozen evaluation records, and dated implementation diary are not reproduced here. Any future public claims should cite a reproducible released evaluation artifact rather than private-only evidence.

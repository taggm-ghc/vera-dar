# Evaluation and findings

[← Project overview](../README.md)

VERA-DAR's principal evaluation question is whether evidence-grounded context construction results in materially better research answers than an otherwise comparable model answering directly. The available experiments are informative but do not establish that claim.

## Historical TRACE comparison (HISTORICAL; not the current system)

This is the canonical copy of the TRACE v1 results; other pages link here. The results are HISTORICAL: they were run on an earlier build (0094955) with an abstract-grounded variant, and they do not describe the current full-document system. The Trace Eval page (`GET /trace/public`) shows the same historical aggregate with its method and limitations.

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

A separate evaluation of a later abstract-grounded `/ask` configuration (since superseded: the current system answers only from admitted full-document passages, and abstracts are not used as evidence; see [Default behaviour](api.md#default-behaviour)) reported 19 of 30 questions answered with checked citations, zero unresolvable citations among 57, and 9 of 105 checked claims classified as unsupported (8.6%, with reported Wilson 95% interval 4.6%–15.5%) before removal from the displayed answer.

**This is not a before-and-after estimate.** That later run did not have a direct-model baseline evaluated under the same claim-check procedure. It therefore does not prove a gain attributable to the retrieval and checking changes.

## Scope-router fix (offline layer-1 measurement)

*Superseded in part 2026-11-05: the pre-retrieval router is now safety-only (no fixed Scope; subject matter is never a reason to decline), so the topic-versus-intent framing below is history. The figures remain a dated layer-1 measurement.*

This counts how the layer-1 rule set of the scope router (`config/scope_router.json`) routes questions; it is **not an answer success rate**. The earlier rules blocked topic words such as "medicine" or "stocks"; the fix replaces topic blocking with intent-based rules. Measured on the 85-probe set the fix was developed against (48 that should pass, 37 that should be blocked; not held out), layer 1 only, reproduced independently twice:

| Measure (offline layer-1 scope router) | Before | After the fix |
|---|---:|---:|
| Benign questions wrongly blocked | 34/48 | 0/48 |
| Should-block questions caught | 28/37 | 36/37 |

Limits: layer 1 only, no model calls; the probe set was written before the fix, was used while developing it (three probes appear in the tests), is not held out and has not been externally reviewed; one operational-harm probe still passes layer 1 and relies on layer 2. Real-model layer-2 behaviour under the new prompt is untested (assumed).

## Status note (as of 2026-10-30; re-checked 2026-11-05)

**No current-system measurement exists.** No before/after measurement of the current full-document pipeline has been made, and no evaluation of the current discovery-to-answer path has been published. Not measured: answer quality of the current full-document path, any comparison with other systems, the accuracy of the claim checker, and live production behaviour of discovery-to-answer acquisition. The figures above are dated history (2026-10-04 and a later run); they were not re-measured in this documentation pass. Offline release-verification runs under `work/RELEASE-VERIFY-1104/` (for example result-20261105T072514Z.md: PASS on offline unit and integration tests and static scans, no live behaviour) show the suites pass; they are not quality measurements. Test-suite counts reported elsewhere are branch- and revision-specific; record the command, revision and counts rather than quoting one number.

## Remaining research questions

- How much does improved retrieval alter answer quality rather than only citation presence?
- Can provenance be tied to inspected full-document passages instead of abstracts alone?
- What is the independent error rate of the claim checker?
- Do benefits persist on genuinely external questions, independent graders and strong current baselines?
- What are the latency, cost and reliability trade-offs of added evidence processing?

Internal trace files, frozen evaluation records, and dated implementation diary are not reproduced here. Any future public claims should cite a reproducible released evaluation artifact rather than private-only evidence.

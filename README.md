# VERA-DAR

**Verifiable Evidence-based Research Answers**

VERA-DAR is an experimental, provenance-grounded research system developed for the **TAI Labs Agentic AI Engineering Bootcamp** capstone. It investigates whether deliberately selecting and examining evidence before generation can produce research answers that are both **more defensible** and **better** than direct responses from leading language models.

**Project status:** Working FastAPI research endpoint and Streamlit interface; active capstone development. The central claim of superiority over leading-model baselines **has not been demonstrated**.

## Purpose and status

VERA-DAR's foundational requirements are full-document acquisition, inspection, evidence evaluation, context construction, answer synthesis and claim verification. Evidence-grounded answer synthesis is the intended primary result of `/ask`; a reading list and an explicit insufficient-evidence result are its defined fallbacks.

**Verified today** (checked against the code, not only the design):

- By default `/ask` returns a reading list: no answer is generated and no claim check runs.
- The verified full-document answering path exists in the code but is disabled (`fulldoc.enabled` is false), pending calibration of its thresholds and a recheck against a real document bundle.
- Automated full-document acquisition is designed but not yet implemented.

## The problem

Language models can produce persuasive answers without exposing how supporting evidence was found, selected, or interpreted. Adding citations alone does not establish that sources were examined adequately, or that the answer is correct.

VERA-DAR investigates a more demanding question:

> Can deliberate, auditable evidence acquisition, critical evaluation, and context construction cause a frontier model to produce a materially better research response than it produces directly?

Success requires two things: **traceable evidence decisions** and **a demonstrably better final answer**. Neither substitutes for the other.

## Approach

By default, the public `/ask` workflow is a **discovery service**. For an accepted research question it searches scholarly indexes and the web and returns a **reading list** of published documents that may be relevant, with links. It generates no answer and runs no claim check on that list, because VERA-DAR has not read the full documents and does not treat abstracts or snippets as evidence.

A grounded, cited answer is produced only on an **optional verified full-document path**: documents are acquired and verified, relevant passages are admitted, an answer is generated from those passages, and cited claims are checked. This path is enabled only when `fulldoc.enabled` is true in config/fulldoc_answer.json and a verified bundle is configured via VERA_FULLDOC_BUNDLE_DIR.

```text
Research question
       |
       v
Scope check and evidence discovery
       |
       +--> default: reading list with links (no generated answer)
       |
       +--> optional, off by default: verified full documents
                   |
                   v
            Admitted passages -> cited answer -> claim check
```

The broader research design also calls for critical evidence evaluation, deliberately constructed context, and comparative assessment. Those ambitions are not all implemented.

## Current capabilities

- **Evidence discovery:** searches arXiv, OpenAlex and the web and returns a reading list of candidate documents.
- **Provenance:** returns source metadata such as provider, identifier, retrieval time, and recorded licence decision.
- **Optional grounded answers:** with the full-document path enabled, cited answers and a claim-check summary.
- **Operational visibility:** API health, version information, usage tokens, and estimated cost.
- **Interactive demonstration:** Streamlit interface for questions and selected evaluation views.

VERA-DAR is a prototype, not an independent verification authority. Search relevance, model-based checking and rights handling have known limitations.

## Why provenance matters

A bibliography is not an evidence trail. Useful provenance should help a reviewer distinguish:

- **Discovery:** where a candidate research record came from.
- **Inspection:** what material was actually read or processed.
- **Use:** which source excerpt was supplied for a generated claim.
- **Assessment:** what checking occurred and what uncertainty remains.

By default the system reports discovery provenance only. Inspection of complete documents exists only on the optional full-document path. That boundary is central to the research design, not a minor documentation detail.

## Evaluation: what has been measured

A small TRACE comparison evaluated an earlier question-only baseline against an abstract-grounded variant, using the same model and frozen checks. It found improvements in some citation-related measures, **but not in the overall rate of passing every applicable check**.

| Measure | Direct baseline | Abstract-grounded variant |
|---|---:|---:|
| Citation present (22 applicable cases) | 1/22 | 10/22 |
| No phantom evidence (22 applicable cases) | 4/22 | 17/22 |
| All checks passed, development set (20 questions) | 2/20 | 3/20 |
| All checks passed, held-out set (10 questions) | 2/10 | 2/10 |

These are **historical, small-sample results** from a variant that used abstracts, which the current default workflow no longer does. They are not evidence that the current application outperforms frontier research tools. See [Evaluation and findings](docs/evaluation.md) for methods and caveats.

## Try it or explore the code

No public demonstration URL is asserted here. The project includes:

- FastAPI backend with `POST /ask` and `GET /health`.
- Streamlit research and demonstration interface.
- Tests and offline evaluation utilities.

For local setup, see [Development](docs/development.md). For service configuration, see [Deployment](docs/deployment.md).

## Privacy and disclosure

Every page of the demonstration shows one shared data-use disclosure. Its closing text reads:

"Your question is sent to third-party AI and search services to generate a response. These services have their own data-handling and retention policies. VERA-DAR does not save your question or information identifying you. During this demonstration, VERA-DAR does not save newly discovered documents or research notes from your questions."

The final sentence applies only while corpus admission and discovery notes are off; if either is enabled, the disclosure text changes accordingly.

When the full-document path is enabled, the disclosure opens differently: it says answers are written from up to six passages of a small, operator-supplied, licence-checked collection of full documents, and that the question, the selected passages with their titles and links, and draft answer sentences are sent to third-party AI services. The closing text quoted above is unchanged.

External provider retention is outside VERA-DAR's control. See [API and data use](docs/api.md).

**OpenAI fallback switch.** `openai_fallback.enabled` in `config/ask-provider-chain.json` is `false` by default. While it is false, the API never calls OpenAI and does not need `OPENAI_API_KEY`; answering and claim checking use Groq only. When it is true, OpenAI is the last fallback after the other providers fail, it needs a valid key, and it is a paid route.

## Important limitations

- **Full documents are not read by default.** An optional verified full-document path exists and is disabled by default; otherwise results are links to published documents, and abstracts or snippets are not used as evidence.
- **Source provenance is incomplete.** Returned index records do not provide immutable document versions or precise passage locations.
- **A source's metadata licence does not grant rights to the underlying paper.** Licence decisions are as reported by indexing services and are not verified with rights holders.
- **Claim checking is imperfect.** Its error rate has not been fully established.
- **Comparative superiority is unproven.** The full multi-stage research pipeline has not been validated end to end against live services.

## Documentation

| Resource | Contents |
|---|---|
| [Documentation index](docs/README.md) | All published documents |
| [API and data use](docs/api.md) | Public endpoints, response semantics, privacy and provenance limits |
| [Development](docs/development.md) | Local requirements, startup and testing |
| [Deployment](docs/deployment.md) | Docker and Render setup for separate API and UI services |
| [Architecture](docs/architecture.md) | Public conceptual architecture and known gaps |
| [Evaluation and findings](docs/evaluation.md) | TRACE methodology, measured results and limitations |

## Project and attribution

**VERA-DAR — Verifiable Evidence-based Research Answers**  
TAI Labs Agentic AI Engineering Bootcamp capstone · 2026

This repository is published for educational demonstration, evaluation, and portfolio purposes. Public access does not imply that undisclosed internal mechanisms, algorithms, or decision methods are described here.

Copyright © 2026 Tagg Maiwald. All rights reserved.

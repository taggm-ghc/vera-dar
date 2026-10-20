# VERA-DAR — Verifiable Evidence-based Research Answers

Status: see the root README.


**Provenance-grounded agentic context engineering for auditable research synthesis**

Prepared September 2026

> This is the public design overview. It describes the text evidence-selection,
> critical-appraisal, reasoning, synthesis, evaluation, and provenance architecture
> implemented for the capstone.

---

## Capstone one-liner

VERA-DAR is a provenance-grounded agentic context-engineering system that discovers, critically evaluates, registers, and organizes evidence into an auditable reasoning context. Its goal is research responses measurably better than direct leading-LLM baselines; that comparison has not yet been run.

## North star

The goal is to demonstrate that the system produces a grounded research response significantly superior in quality to direct responses to the same research question from leading frontier LLMs, while making its evidence-selection process inspectable and auditable. Every consequential evidence decision should be traceable to its source, evidence, provenance, and evaluation method.

## Design rationale

VERA-DAR's governing proposition is not "select defensible evidence" but "create demonstrable incremental research value." A well-grounded, auditable response that is materially worse than a direct frontier-model response fails the outcome test.

The architecture therefore treats critical inquiry, evidence-set adequacy, explicit inference, synthesis, answer evaluation, and bounded revision as first-class system stages rather than leaving final answer quality to the generating model.

## Scope

VERA-DAR is deliberately bounded to text. It does not address multimodal representations, universal ontologies, production malware analysis, or generalized autonomous research. Search, fetch, parsing, and model inference are treated as reusable tools; the engineered contribution is the reasoning context constructed between them. PostgreSQL is the canonical memory and audit store.

## Research foundations

| Research source | Transferable principle | How VERA-DAR applies it |
| --- | --- | --- |
| Facione critical-thinking framework [1] | Interpretation, analysis, evaluation, inference, explanation, and self-regulation are separable critical-thinking operations. | Explicit inference, explanation, and self-evaluation stages after evidence appraisal. |
| ACRL Framework [2] | Research is iterative inquiry; searching is nonlinear strategic exploration; authority is contextual; information creation is a process. | Evidence-needs decomposition, search iteration, process appraisal, and search-again behavior. |
| Cochrane search guidance [3] | High-recall search is preferable when missing relevant evidence would distort conclusions; multiple reports may derive from one underlying study. | Gate A biased toward recall; evidence lineage and independence tracking. |
| Wang & Strong data quality [4] | Data quality includes intrinsic, contextual, representational, and accessibility dimensions and is task-dependent. | Data quality, evidentiary quality, and applicability kept separate rather than one weight score. |
| CASP critical appraisal [5] | Trustworthiness, results/value, and relevance require systematic appraisal; venue or author alone is not enough. | Source/process and methodology appraisal with explicit unknown states. |
| GRADE/Cochrane evidence-body concepts [6] | Bodies of evidence require assessment of inconsistency, indirectness, imprecision, bias, and missingness. | Gate C evidence-set adequacy rather than judging only atomic passages. |
| Wineburg & McGrew lateral reading [7] | Professional fact-checkers verify unfamiliar sources using external sources rather than relying on the site itself. | Lateral source verification for consequential evidence. |
| Sperber et al. epistemic vigilance [8] | Source credibility and content/argument acceptability are distinct vigilance problems. | Source credibility, claim support, and inferential validity assessed separately. |
| PRISMA flow principle [9] | Identification, exclusion, inclusion, and reasons should remain traceable. | Candidate and exclusion histories preserved with reasons, without claiming PRISMA compliance. |

## Capstone seed

| Question | Answer |
| --- | --- |
| What does it do? | Discovers candidate evidence, identifies what evidence the question requires, selects which sources warrant acquisition and acquires and registers the complete document of each selected source (target; see "Current state vs required"), critically appraises claims and source processes, models support and contradiction, determines whether the evidence set is adequate, constructs an auditable reasoning context, and generates and critiques a grounded response. |
| Who is it for? | A researcher using AI to assemble and reason over an auditable evidence base for a bounded research question. |
| What must it handle well first? | Given a bounded research question, what evidence is needed, which candidate evidence is worth acquiring, what evidence is trustworthy and applicable, whether the evidence set is sufficient, and what conclusions are warranted by it? |
| What would make a proud demo? | A grounded response measurably superior to direct leading-LLM baselines, with every consequential evidence and reasoning decision inspectable and traceable to source, evidence, provenance, method, and uncertainty. |

## Success and failure conditions

| Condition | Pass / fail rule |
| --- | --- |
| Response quality | PASS only if the capstone response materially exceeds a predeclared direct-model baseline on the evaluation dimensions that matter for the question. |
| Grounding | FAIL if consequential claims cannot be mapped to admitted evidence or if citations do not support the claims they accompany. |
| Reasoning integrity | FAIL if conclusions do not follow from evidence, material counterevidence is ignored, or uncertainty is hidden. |
| Auditability | FAIL if consequential acquisition, appraisal, admission, inference, exclusion, or revision decisions cannot be traced. |
| Incremental value | FAIL if the pipeline adds cost, latency, and complexity yet yields response quality no better than or worse than direct frontier-model baselines. |

> Demo success = superior response quality **AND** grounding **AND** reasoning integrity **AND** auditability. None of these compensates for failure of another.

## System boundary and workflow

The differentiated capability is context engineering, extended to include the critical-thinking operations required to turn evidence into a superior answer.

1. Accept a bounded research question and governing context.
2. Interpret the question and create an Evidence Requirements Map that specifies material subquestions, evidence types, temporal scope, authority needs, counterevidence needs, and coverage state.
3. Plan one or more search strategies and discover candidate sources.
4. **Gate A:** prioritize candidates before full acquisition using query fit, expected evidentiary value, source cues, novelty, cost/risk, and uncertainty, while favoring recall over premature exclusion.
5. For each selected candidate, resolve the original document and version and acquire the complete accessible document (`full_document`) into quarantine/staging; compute content hash and register source/version/provenance and acquisition status before semantic use (see "Full-document acquisition"; target, see "Current state vs required").
6. Run bounded deterministic content-integrity checks and preserve exact finding spans.
7. Segment relevant content into semantic evidence units with exact source locations.
8. Critically appraise source/process, data quality, evidentiary quality, applicability, temporal validity, independence, uncertainty, meta-drift, and reference-relative bias where supported.
9. Perform lateral verification for consequential or unfamiliar sources and record external source-appraisal evidence.
10. **Gate B:** admit, qualify, reject, quarantine, or excise-span-and-admit individual evidence.
11. Build an evidence-relation and argument layer linking claims to supporting, contradicting, qualifying, and derivative evidence, explicit assumptions, and alternative interpretations.
12. **Gate C:** evaluate the evidence set as a whole for coverage, directness, independence, contradiction, currency, missingness, uncertainty, and diminishing-return saturation.
13. If Gate C fails adequacy, formulate the missing evidence need and iterate search rather than forcing synthesis.
14. Construct model context from evidence and argument structures, preserving provenance, qualification, counterevidence, and uncertainty.
15. Generate a deliberate draft synthesis.
16. Critically evaluate the draft for completeness, grounding, inferential validity, contradiction handling, qualification, scope fidelity, relevance, synthesis, and clarity.
17. Perform one bounded revision when material defects are detected.
18. Return the grounded response and persist decisions, findings, evidence links, answer evaluation, comparative baseline results, and feedback.

## Three decision gates

| Gate | Question | Primary output |
| --- | --- | --- |
| Gate A: acquisition priority | Is this candidate sufficiently likely to add relevant evidence to justify acquisition or later review? | `fetch` / `defer` / `reject`, with uncertainty and factors |
| Gate B: evidence admission | After provenance, integrity, critical appraisal, drift, epistemic and bias evaluation, may this evidence participate in reasoning? | `admit` / `qualify` / `reject` / `quarantine` / `excise-span-and-admit` |
| Gate C: evidence-set adequacy | Does the admitted evidence set cover the material needs of the question well enough to warrant synthesis? | `synthesize` / `search-again` / `answer-qualified` / `insufficient-evidence` |

Gate A controls acquisition resources, Gate B protects reasoning context, and Gate C protects answer adequacy.

## Evidence Requirements Map

Before search, VERA-DAR creates a machine-addressable representation of what would have to be established to answer the governing question well. This prevents high-volume retrieval from being mistaken for evidence completeness.

| Field | Purpose |
| --- | --- |
| Subquestion | Separates the governing question into material evidentiary needs without prematurely answering them. |
| Claim type | Identifies whether the requirement concerns fact, causal claim, comparison, forecast, definition, normative judgment, or another bounded type. |
| Desired evidence type | Guides search toward evidence capable of answering the requirement rather than merely mentioning the topic. |
| Temporal scope | Defines freshness and historical coverage required by the question. |
| Counterevidence need | Requires active search for evidence that could weaken or reverse a provisional conclusion. |
| Coverage state | `unsearched` / `partial` / `adequate` / `contradictory` / `unresolved` / `not-obtainable` |

## Search strategy and Gate A

Pre-acquisition scoring is a prioritization mechanism, not a cost-control shortcut: a Gate A false negative can permanently remove decisive evidence from the reachable evidence universe.

For bounded research, Gate A should usually prioritize recall over precision — important uncertain candidates may be deferred or fetched rather than permanently rejected. The pre-acquisition estimate remains a prioritization estimate, not a finding about acquired content.

| Signal | Role |
| --- | --- |
| Query/context fit | Estimate topical and task fit from title, snippet, URL, metadata, and available context. |
| Expected evidentiary value | Estimate whether the candidate is likely to contain material evidence, not merely topical language. |
| Source/provenance cues | Use publisher, author, domain, date, and source type as contextual cues, not proof of authority or bias. |
| Novelty / redundancy | Prefer candidates likely to add new evidence or independent corroboration. |
| Coverage contribution | Estimate which open Evidence Requirement the candidate may satisfy. |
| Acquisition cost/risk | Account for fetch cost, parsing difficulty, duplication, malformed content, and obvious unusability. |
| Uncertainty | Preserve how little is known before acquisition and avoid converting uncertainty into rejection. |

## Full-document acquisition (required architecture, owner ruling 2026-10-14)

The required acquisition target is `full_document` for every source selected for evidentiary acquisition. It is the required target, not an optional future step and not a target applied only when convenient. Acquisition is selective about **which** sources warrant retrieval (Gate A). It is not selective about **whether** a selected source's complete document is acquired. Discovery metadata, index records, snippets, summaries and abstracts help evaluate candidates and set acquisition priority. They are not substitutes for the source document.

### Pipeline stages

1. **Discovery.** Search providers return candidates as snippets, index metadata, abstracts or summaries.
2. **Candidate evaluation and selection.** Gate A uses those lesser representations to choose which candidates to acquire (`fetch` / `defer` / `reject`). The Gate A decision and its reason are recorded for every candidate; only `fetch` candidates enter acquisition.
3. **Resolution of the original document and version.** For each selected candidate, resolve the original document (not an index page, mirror or portal) and its specific version (for example a DOI, an arXiv version, or a publisher version of record).
4. **Acquisition (to staging).** Acquire the complete accessible document, subject to applicable rights and access controls, into quarantine/staging. Record source identity, provenance, acquisition status (with a failure reason when it fails), version type, version identifier and content hash before any semantic use.
5. **Rights assessment.** Record each licence record with its provenance and scope, and the resulting rights disposition. The document's own licence statement may be read from the staged copy. Content leaves quarantine/staging only when the disposition is `permitted` (see Storage and licence); otherwise the disposition is recorded and the staged content is not used. A rights assessment may also run before stage 4, on licence records available from discovery or resolution. If it already finds that the disposition does not permit staging, stage 4 is skipped and the acquisition status is `blocked_rights`.
6. **Inspection and evaluation.** Run integrity checks and critical appraisal on the acquired document before any of its contents are admitted as substantive evidence. Record the inspection outcome and the representation type actually found.
7. **Passage-level evidence admission.** Gate B admits, qualifies or rejects specific passages of the inspected document, each with its exact location.
8. **Citation and claim check against the examined passages.** Each cited claim is checked against the admitted passages it cites, not against an abstract or index record.
9. **Memory gate.** Only findings that passed the claim check against examined passages are offered to the memory write gate (`vera/memory_gate.py`); storage stays with `vera/memory_store.py` and owner confirmation (AGENTS.md rule 2).

### Data model: separate dimensions

Each candidate, and each acquired or attempted source, carries the following dimensions. Each dimension is recorded in its own field. No value is copied from another dimension without the check that sets it: for example, representation type `full_document` is set only by an inspection that `passed`. The separation of dimensions is required (DEC-FDA1). The vocabulary below is **ratified as design baseline 2026-10-14 (OD-1; recorded as DEC-OD1); not implemented**. No fields, API changes or DDL exist or are authorised by this ratification. The gap analysis quotes these lists and defines none of its own; this section is the one canonical vocabulary.

| Dimension | Vocabulary | Definition |
| --- | --- | --- |
| Gate A decision (applies to every candidate) | `fetch` | Selected for evidentiary acquisition. Only these candidates get an acquisition status. |
| | `defer` | Not selected now; may be reconsidered. Recorded with its Gate A reason (today `candidates.gate_a_decision`). It has no acquisition status: it is not `not_attempted`, because acquisition was never due. |
| | `reject` | Not selected. Recorded with its Gate A reason. It has no acquisition status. |
| Representation type (what artifact is held) | `discovery_snippet` | A search-result excerpt or ranking snippet written or cut by the search provider. |
| | `index_metadata` | A bibliographic or index record (title, authors, date, identifiers, venue) from an index such as OpenAlex, with no body text. |
| | `abstract` | The author-written or publisher-provided abstract, whether taken from the document or from an index. |
| | `summary` | A summary written by a third party or a model, not by the author. |
| | `landing_page` | A publisher, repository or DOI landing page about the document, not the document itself (what the M2 path usually stores today). |
| | `unclassified` | An acquired artifact that has not yet been inspected. It is never treated as `full_document`. |
| | `full_document` | The complete accessible body of the resolved original document, at an identified version. Assigned only when the inspection outcome is `passed`. |
| Acquisition status (what happened when VERA-DAR tried to acquire the full document; Gate A `fetch` candidates only) | `not_attempted` | The source was selected, but no acquisition has been tried yet. |
| | `in_progress` | Acquisition has started and has not finished. |
| | `acquired` | Bytes were retrieved in full, hashed and staged. It says nothing about whether the bytes are the document; that is the inspection outcome. |
| | `failed` + reason | Acquisition was tried and did not succeed. A reason is required: `not_resolved`, `timeout`, `fetch_error`, `too_large` (the body exceeds the byte cap; the cap is a threshold and needs a calibration note), `unsupported_format` (for example a PDF while no PDF parser exists), `parse_error`, `truncated` (the transfer ended before the body was complete). |
| | `blocked_rights` | Acquisition into staging did not proceed because a rights assessment made before acquisition found that the disposition does not permit even staging. The disposition (`restricted`, `unresolved` or `conflicting`) is recorded separately and is the reason. This status records only that acquisition stopped. It is not a rights finding. |
| | `blocked_access` | The document exists but cannot be reached, for example a paywall, login, bot challenge or refused request. VERA-DAR never bypasses an access control. |
| Inspection outcome (what inspection found in an `acquired` artifact) | `not_inspected` | Acquired, not yet inspected. Not admissible. |
| | `passed` | Integrity checks and appraisal passed: the artifact is the resolved document, at the expected version, complete. |
| | `failed` + reason | A reason is required: `not_the_document` (a landing page, abstract page, paywall or cookie wall), `wrong_document` (title or identifier does not match the candidate), `version_mismatch` (a different version from the one resolved), `incomplete_body` (body sections missing), `unreadable` (no usable text extracted). The representation type records what was actually found, for example `landing_page`. |
| Rights and licence provenance (who says what the licence is) | `verified_at_source` | The licence was checked with the rights holder or at the original document (publisher page, the repository's licence record for the exact deposited version checked against that version, or the document's own licence statement). |
| | `asserted_by_intermediary` | The licence is stated only by an index or aggregator. This is an assertion, not a verification. |
| | `unknown` | No licence information was found. |
| Rights disposition (the outcome of the rights assessment; separate from licence provenance and licence scope) | `permitted` | The assessment finds that the intended acquisition, storage and use of the content is allowed. |
| | `restricted` | The assessment finds that the terms forbid or limit the intended acquisition, storage or use. |
| | `unresolved` | The assessment has not reached a finding, for example because it is incomplete or because licence provenance is `unknown`. |
| | `conflicting` | Sources of licence information disagree. It never resolves to `permitted` without verification at source. It changes only after a check at source: to `permitted` if the at-source terms permit the intended use, otherwise to `restricted`. The conflicting records and the resolution are kept. |
| Licence scope (attribute of each licence record) | `metadata`, `abstract`, `content` | What the licence record covers. A `metadata` or `abstract` licence never clears `content`. |
| Version identity (named dimension; made of the two attributes below) | Version type plus version identifier | Which version of the work was acquired. Kept distinct from acquisition, representation, inspection, licence provenance, rights disposition and scope. |
| Version type (attribute) | `preprint`, `accepted_manuscript`, `version_of_record`, `unknown` | The kind of version acquired. Distinct from the version identifier. Which types are acceptable as evidence is OD-2. |
| Version identifier (attribute) | Identifier plus version | The specific version acquired: DOI, arXiv id with version suffix or other version-specific identifier, plus the retrieval timestamp and resolved URL. Different versions are different source versions. Distinct from the database's storage version number (`sources.version`). |
| Content hash | Hash of acquired bytes and of extracted text | Computed when the document is staged. Original acquired content is immutable (Provenance invariant); any change produces a new version record. |
| Quoted passages with locations | Passage text plus location | Each passage admitted or quoted has its exact text and its location in the hashed version (character offsets, plus page, section or paragraph where available). Quotation invariant limits apply. |

**Proposed, not adopted (owner decisions pending, 2026-10-14).** Two amendments to the attributes above are proposals only; the DEC-OD1 dimensions and values stay as ratified. (OD-12c) Record the extractor name and version with the content hash, because a different extractor version gives a different extracted-text hash; completeness would stay recorded by the acquisition status `truncated` and the inspection outcome `incomplete_body`, not by the hash. (OD-12b) Add an "intended operation" attribute to each rights assessment (`acquire`, `retain`, `index`, `embed`, `model_context`, `quote`), with one disposition per operation; the existing scope rule (a `metadata` or `abstract` licence never clears `content`) is unchanged. OD-12b also asks whether an index's own metadata licence counts as `verified_at_source` for scope `metadata`. If adopted, the `permitted`, `restricted` and `conflicting` disposition rows would name "the operation the assessment names" in place of the current wording. Neither is applied to the table until the owner decides.

**Why one combined enumeration is inadequate.** These dimensions vary independently. An abstract can be held because acquisition is `not_attempted`, `failed`, `blocked_rights` or `blocked_access`, and each needs a different follow-up. A file can be `acquired` and still fail inspection as a `landing_page`. A `full_document` can be `acquired` with a licence that is `verified_at_source` or only `asserted_by_intermediary`, and with any of the four rights dispositions. Only a licence record of scope `content` that is `verified_at_source`, together with disposition `permitted`, lets it be stored beyond staging or used. A single value such as "abstract / full text / failed / blocked" cannot say what was held, why the full document is missing, or whether the licence was verified or what the rights disposition is. It hides exactly the facts that the evidence admission and failure rules below must check. A combined enum would also tempt code to treat "best representation held" as success. Version, hash and passage locations are attributes of a specific acquired artifact, not states, so they cannot be values of an enum at all.

### Evidence admission rule

Substantive evidence comes only from passages of a `full_document` whose acquisition status is `acquired`, whose inspection outcome is `passed`, whose licence record of scope `content` has provenance `verified_at_source`, and whose rights disposition is `permitted`. The last two are independent, and both are required: `restricted`, `unresolved` and `conflicting` dispositions stop admission, and the stop is recorded with its disposition. Discovery snippets, index metadata, abstracts and summaries are used only for candidate evaluation, selection and acquisition priority. They are never admitted as substantive evidence, and claims are never grounded or checked against them. Quotations and citations point to the examined passage locations in the hashed version.

### Claim-to-passage-to-version chain (acceptance criterion)

Every substantive citation or quotation, in `/ask` answers, memory rows, M3 to M5 outputs and evaluation reports, must satisfy all of the following:

1. The claim resolves to one or more passage records.
2. Each passage resolves to exactly one immutable source version (a `sources` row), never only to a canonical source.
3. The recomputed hash of that version's content equals its recorded content hash.
4. The passage text equals the version's content at the recorded offsets.
5. The version satisfies the evidence admission rule above.
6. The claim check was made against those passage texts, not against an abstract, snippet or index record.
7. The whole chain can be rebuilt from database records alone.

Proposed (OD-12a, not adopted until the owner decides): the admission rule would be implemented as one admission predicate over the recorded dimensions, called by `/ask` citation building, the M3 entry, the memory write gate and owner confirmation. Its result would be derived when needed and not stored as a separate field (DEC-EI1: derived, never asserted). Under that design a source that fails the predicate would not be used as evidence, and the reason would be recorded. The chain requirement above is REQUIRED; the single-predicate design is proposed; neither is implemented (REQUIRED, DESIGNED, NOT IMPLEMENTED).

### Evidentiary-inspection ceiling (required, owner ruling DEC-EI1, 2026-10-14)

Owner ruling (verbatim): "No substantive source citation or quotation may claim a level of evidentiary inspection greater than the source content actually acquired, verified, and examined."

- **Derived, never asserted.** The evidentiary level that any citation or quotation displays is derived from the recorded acquisition status, representation type, licence provenance and scope, rights disposition and inspection outcome, plus whether the cited passage was located in the hashed version. It is never set independently: not from a provider literal, a code path's assumption, a default value or a model's statement.
- **The ceiling is the minimum.** The displayed level is the minimum of what was acquired, what was verified and what was examined. A `full_document` that is `acquired` but `not_inspected` shows as not examined. A full document whose content licence is not `verified_at_source`, or whose rights disposition is not `permitted`, cannot be shown as used evidence. A claim checked against an abstract shows as checked against an abstract, not as verified.
- **Scope.** VERA-DAR's answers, API fields, UI labels, claim-check summaries, memory rows, logs, evaluation reports, docs and presentation material. Words such as "fetched", "full text", "full content", "verified", "original" and "supported" are used only when the records support them.
- The derivation is a function over the recorded dimensions, computed when the label is shown. It is not a stored field and not a combined enumeration.
- **Explicit label (owner ruling DEC-R9d, 2026-10-14; surface list per reconciliation item CX-4).** Every substantive citation carries an explicit inspection-scope label computed by this derivation. This covers API sources and traced sources, UI source lists, memory recall and owner review lists, claim-check and verification reports, and evaluation files. The label is never inferred upward from a provider name, a source identifier or an existing static label such as `stored_text_kind`; such fields may only lower it. Where no acquisition record exists, the label is the lowest level the records support, for example "abstract or index record; full document not acquired". Existing records are not rewritten; their current display carries the label (DEC-R9c). The deterministic ceiling test joins the release gate once implemented (DEC-R9e). Until then, Rule 9 is NOT COVERED by the release gate.

### Failure rule

When full-document acquisition fails or is blocked, when inspection fails, or when the rights disposition is not `permitted`, VERA-DAR records the outcome with its status, reason and disposition. It never presents metadata, a snippet, a summary or an abstract as acquired full-document evidence. For each source, the API response and the UI show the representation type actually used, its acquisition status, its inspection outcome and its rights disposition. A source without an acquired full document is shown as not used as evidence, or the answer is qualified or marked as insufficient evidence through Gate C.

### Owner confirmation: revalidation and conditional atomic transition

Owner confirmation makes a pending (or legacy pre-D6) memory finding recall-eligible. It is an integrity decision as well as a protected status transition. Before the transition, the service re-checks, from database records only and in one read-only snapshot:

1. the row's state (`pending` or `legacy_auto`) and that it has not expired;
2. that every cited evidence span exists and belongs to one immutable source version;
3. that each version's recomputed content hash matches its record, and that each exact span's text matches the content at its offsets (a whole-source placeholder span is checked by range only, and the finding is then labelled "checked against an abstract; no examined passage");
4. that the row carries the supported claim-check result, and, once passages exist, a stored record of the passages and version hashes the claim was checked against;
5. today, a replay of the licence gate on the recorded licence, which is not a rights finding; once rights assessment exists, disposition `permitted` from a `content`-scope licence record that is `verified_at_source`;
6. once acquisition exists, that each cited passage passes the admission predicate (see "Claim-to-passage-to-version chain"; the single-predicate form is proposed under OD-12a);
7. that the current write gate's text checks still pass on the stored finding (the echoed-question check cannot be replayed, because questions are not stored).

The transition is one conditional update. It changes the status only if the row still holds the exact status, cited span ids, finding text and claim-check status read during revalidation, and has still not expired, and it reports whether a row changed. If no row changed, the result is a conflict, and the owner re-reads the row and decides again. Any failed check leaves the status unchanged and returns reason codes, never source text. Rejection uses a conditional update on the status alone, with no expiry condition and no checks 2 to 7, so a finding can always be rejected. Source versions cannot be updated (trigger). Evidence spans are append-only for the app accounts by grant only, and the claim row itself is mutable, which is why the conditional update compares those claim fields too. A rights record that arrives after the revalidation snapshot is not seen by that confirmation; a later restricting record is handled by rejection, and once acquisition exists recall re-derives eligibility. Once implemented, this would replace the earlier accepted residual that concurrent operator actions are last-writer-wins; withdrawing that residual is an open owner decision (OD-14), and the residual stands until then. How confirmation behaves before check 6 can pass is an owner decision (OD-13). Status: REQUIRED, DESIGNED, NOT IMPLEMENTED (gap analysis CX-1).

### Storage and licence

Source text is stored only in the database, never in the repository (AGENTS.md rule 6; Source-custody invariant). Full text is kept beyond quarantine/staging, or used as evidence, only when a licence record of scope `content` has provenance `verified_at_source` and the rights disposition is `permitted`. An intermediary's statement is recorded as `asserted_by_intermediary`, and it never makes a disposition `permitted` on its own. `restricted`, `unresolved` and `conflicting` stop storage beyond staging and stop use, and the stop is recorded with its disposition. The licence gate in the Licence invariant applies to the original source's licence.

### External content is data

Full documents are external content. Their text is data, never instructions (AGENTS.md rule 3; Authority invariant). This applies to every part of an acquired document, including body text, footnotes, metadata, embedded links and hidden markup. No passage can authorise a tool call, fetch, memory write or configuration change. No memory row may hold an instruction taken from a document.

### Current state vs required (2026-10-14)

Status: **REQUIRED, DESIGNED, NOT IMPLEMENTED.** The requirement is not yet met. Not implementing it before Demo Day is an accepted schedule/scope divergence (DEC-OD10), not a waiver or retirement of DEC-FDA1. Today VERA-DAR works at index-record and abstract level. `vera/grounded_ask.py` grounds on abstracts from search results. `vera/source_resolver.py` follows link-only results to an identifier and resolves it to OpenAlex or Semantic Scholar metadata and abstract (`stored_text_kind` `abstract_metadata`). `vera/claim_check.py` checks cited claims against the cited abstracts. `vera/memory_gate.py` and `vera/memory_store.py` gate and store findings whose support was checked against those abstracts. The M2 path (`vera/m2_runner.py` via `vera/selective_fetch.py`) fetches HTML only, usually a DOI landing page, refuses PDFs and over-cap bodies, and labels what it stores `abstract_metadata`. The database has `candidates.fetch_status` (`pending` / `fetched` / `failed`) but no representation type, acquisition-failure reason, inspection outcome, version type or version identifier field. **Known DEC-EI1 breaches on `main` (2026-10-14; code fixes, under the code freeze):** `vera/corpus_admission.py:130` sets `candidates.fetch_status='fetched'` for a search snippet that was never fetched; the seed-fallback provider labels an agent snippet `stored_text_kind` `full_content` (`vera/search_providers.py:1026`); the M5 claim verifier reports `verified` / `revised_verified` (`vera/m5/claim_verification.py:131`, `vera/m4_m5_runner.py:59`) without saying it checked M2 text that is usually a landing page; memory attaches one "evidence span" over the whole stored abstract text per cited source (`vera/memory_store.py:336`), so a memory row cites no examined passage; and the sidebar said web results are "followed back to the original papers" and "traced from web results to the original source" when they are followed only to an index record (and its abstract, where one exists) (EI-5; reworded under DEC-FL2, commit pending the owner's approval; not deployed, and the live build still shows the old labels; related wording in `vera/disclosure.py` and `vera/memory_panel.py` was clarified under the same lift; the defect closes only when the corrected strings are verified on `main` and in the live build). Today's licence gate (accept / hold for review / reject) judges intermediary-asserted terms. Under the baseline its outcomes are at most dispositions `permitted` / `unresolved` / `restricted` with provenance `asserted_by_intermediary` or `unknown`, never `verified_at_source`, so none meets the evidence admission rule. Licences recorded for abstracts are often intermediary assertions (for example OpenAlex's metadata licence) or unknown. Abstract-level grounding is the current state, not the intended end state. The gap analysis is kept in the owner's planning records outside this repository. Any implementation remains subject to the existing code freeze and the owner's approval controls; this section authorises no code or deployment.

Two further gaps are recorded (Codex audit reconciliation, DEC-CX1). First, owner confirmation (`POST /memory/{id}/confirm`) is today a protected status transition only. It checks the row's state and expiry, then updates the status in a separate statement. It does not re-check spans, source version or hash, rights or eligibility, and the read and the update are not one atomic conditional operation. Second, the offline M3 stage reads every stored source's content for extraction with no representation or eligibility check, so it can process abstracts and landing pages as if they were documents. Both are recorded in the gap analysis (CX-1, CX-3) and stay under the code freeze.

## Critical appraisal layer

VERA-DAR separates source appraisal, data quality, evidentiary quality, applicability, and inferential support. No single source score is permitted to silently substitute for these dimensions.

| Construct | Representative components | Rule |
| --- | --- | --- |
| Source/process appraisal | publisher/author identity, creation process, methodology visibility, funding/conflict cues, correction/retraction status when applicable | Authority is contextual and process-dependent; source reputation is not claim truth. |
| Data quality | accuracy, completeness, consistency, timeliness, interpretability, accessibility where relevant | Data quality is task-dependent and distinct from evidentiary force. |
| Evidentiary quality | methodological adequacy, directness, measurement validity, precision, replication/corroboration | Assess the evidence supporting the proposition, not only the document containing it. |
| Applicability | population/context fit, temporal fit, scope fit, transferability | High-quality evidence can be inapplicable to the active question. |
| Independence | common dataset, common study, copied reporting, citation lineage, direct replication | Multiple publications do not automatically equal independent evidence. |
| Uncertainty | epistemic, semantic ambiguity, measurement, model/evaluator, missingness | Unknown, unreported, unevaluated, and neutral are distinct states. |

> Governing separation: source credibility ≠ data quality ≠ claim quality ≠ task applicability ≠ inferential validity.

## Lateral source verification

For consequential or unfamiliar sources, VERA-DAR may leave the source and seek independent information about the author, publisher, primary-source relationship, citation lineage, corrections, and external characterization. The result is registered as evidence about the source, not as an intrinsic credibility label.

| Trigger | Possible verification action |
| --- | --- |
| Unfamiliar publisher or author | Seek independent descriptions, organizational records, or primary-source identity. |
| Consequential claim | Locate the cited original study, dataset, filing, standard, court decision, or other primary artifact where feasible. |
| Apparent consensus | Trace whether multiple reports depend on the same underlying source. |
| Conflicting source descriptions | Preserve disagreement and uncertainty rather than choosing by reputation alone. |
| Correction/retraction-sensitive domain | Check for corrections, retractions, superseding versions, or authoritative updates where feasible. |

## Semantic evidence unit and evidence relations

VERA-DAR represents evidence as machine-addressable textual units linked to exact source versions, locations, propositions, research context, and provenance.

Evidence units are the smallest capstone-visible textual evaluands. VERA-DAR adds explicit relationships among them so that the system can distinguish independent support from duplication, contradiction, qualification, and derivation.

VERA-DAR records explicit relationships among evidence and propositions, together with the evaluation method, provenance, and uncertainty state.

| Relation type | Meaning |
| --- | --- |
| supports | Evidence increases warrant for a proposition under the registered context. |
| refutes | Evidence decreases warrant for a proposition. |
| qualifies | Evidence narrows, conditions, or limits a proposition. |
| replicates | Evidence independently repeats a relevant observation or result. |
| derives-from | Evidence/report depends on another source, dataset, study, or transformation. |
| duplicates | Evidence substantially repeats the same underlying informational contribution. |
| contextualizes | Evidence changes interpretation without directly supporting or refuting the proposition. |

## Epistemic qualification

| Structure | Components | Rule |
| --- | --- | --- |
| Charge | support, refutation, uncommitted | Normalize only within a registered proposition-state frame. |
| Weight vector | source/process reliability, methodological quality, applicability, temporal validity, independence, provenance completeness | Preserve components and derivation; do not collapse into an unexplained intrinsic source score. |
| Uncertainty vector | epistemic, semantic, measurement, evaluator/model, missingness | Unknown is distinct from zero, neutrality, and absence. |
| Directness | direct evidence vs inferential/secondary distance | Do not allow proximity to a source to substitute for evidence quality, or vice versa. |
| Coverage contribution | which Evidence Requirement is addressed and to what extent | Relevance to one requirement does not imply adequacy for the question as a whole. |

## Atomic bias intersection and refusal contract

Bias is reference-relative and is not inferred from source affiliation or viewpoint. Each atomic bias assessment addresses one target under an explicit reference-relative contract. Missing, unevaluated, unsupported, or incomparable states are never treated as neutral or encoded as zero.

| Criterion | Question | Reference example |
| --- | --- | --- |
| Selection | Was relevant evidence included or omitted? | Declared candidate/evidence universe |
| Evidentiary weighting | Was consideration disproportionate to evidence quality and applicability? | Registered quality/applicability basis |
| Framing | Did wording systematically alter treatment of the target? | Counterfactual or comparison framing |
| Attribution | Was agency, cause, or responsibility assigned disproportionately? | Registered causal/evidentiary comparison |

> A bias finding requires target, criterion, reference, context, method, evidence, and provenance. Otherwise the system returns an explicit unsupported, insufficient, incomparable, or not-applicable status.

## Two-axis meta-drift

VERA-DAR evaluates **scope displacement** separately from **contextual relevance** so that useful expansion, appropriate narrowing, and genuine irrelevance are not conflated.

Scope displacement is not itself a penalty. Relevant expansion may be valuable; relevant contraction may be precise but incomplete. These dimensions can be evaluated where scope and relevance materially affect evidence selection, reasoning, or answer quality.

## Argument and inference layer

Admitted evidence does not automatically warrant a conclusion. VERA-DAR inserts an explicit argument layer between evidence and final synthesis so that the model must preserve the inferential bridge from evidence to claim and expose material alternatives.

VERA-DAR explicitly associates consequential claims with supporting and contradicting evidence, inferential warrants, assumptions, qualifications, alternatives, uncertainty, and provenance.

| Component | Requirement |
| --- | --- |
| Claim | A bounded proposition the answer may assert. |
| Supporting evidence | Exact admitted evidence supporting the claim. |
| Contradicting evidence | Material admitted evidence that weakens or conflicts with the claim. |
| Inference / warrant | The explicit reason the evidence supports the claim. |
| Assumptions | Unproven conditions required for the inference to hold. |
| Qualifier | Scope, probability, conditionality, or uncertainty appropriate to the evidence. |
| Alternatives | Plausible competing interpretations or hypotheses. |
| Discriminating evidence | Evidence that would distinguish among material alternatives where available. |

> A reputable source does not make a proposition true, and a large number of dependent sources does not create independent corroboration.

## Gate C: evidence-set adequacy

Gate C evaluates the admitted evidence collection as a body rather than treating atomic admissibility as sufficient. It prevents synthesis from beginning merely because some good evidence exists.

| Dimension | Gate C question |
| --- | --- |
| Coverage | Are all material Evidence Requirements represented? |
| Directness | Does the evidence actually address the governing question and claims? |
| Independence | Are apparent corroborations independent or derived from common sources? |
| Contradiction | Have material competing findings and interpretations been surfaced? |
| Source/evidence diversity | Are relevant evidence classes represented rather than one convenient source class dominating? |
| Currency | Is temporal coverage appropriate to the question? |
| Precision | Is the evidence precise enough to support the intended level of conclusion? |
| Missingness | Is important evidence known or plausibly likely to be absent? |
| Uncertainty | Would unresolved uncertainty materially change the answer? |
| Saturation | Is further search producing diminishing informational return relative to cost and time? |

Gate C output = `synthesize` / `search-again` / `answer-qualified` / `insufficient-evidence`. "Search-again" creates a new search iteration tied to the unmet evidence requirement rather than issuing an unconstrained generic search.

## Context-construction policy

The final context is an engineered reasoning package, not a bag of retrieved chunks. It preserves what the generator needs to reason accurately while making provenance and qualification machine-inspectable.

| Context component | Included material |
| --- | --- |
| Question frame | Governing question, scope, temporal boundary, and response requirements. |
| Evidence requirements | Satisfied, partial, contradictory, unresolved, and unavailable requirements. |
| Arguments | Claims, supporting/refuting evidence, warrants, assumptions, alternatives, qualifiers. |
| Evidence excerpts | Only admitted/qualified evidence with exact source spans and provenance IDs. |
| Source/process notes | Only appraisal information material to interpreting consequential evidence. |
| Uncertainty | Relevant uncertainty and missingness attached to claims rather than one global confidence score. |
| Counterevidence | Material conflicting findings retained explicitly to prevent one-sided synthesis. |
| Meta-drift constraints | Scope-congruence and relevance information used to prevent contraction, expansion, and irrelevance in the answer. |

## Deliberate synthesis and critical self-regulation

VERA-DAR treats the first generated answer as a draft evaluand rather than the endpoint. The system performs one bounded critique-and-revision pass when material defects are detected. This operationalizes critical-thinking self-regulation without creating an open-ended self-reflection loop.

| Answer-quality dimension | Evaluation question |
| --- | --- |
| Question coverage | Did the draft answer every material part of the governing question? |
| Claim support | Does every consequential factual claim map to admitted evidence? |
| Inferential validity | Do conclusions actually follow from the cited evidence and registered assumptions? |
| Evidence completeness | Were material admitted findings or requirements omitted from synthesis? |
| Contradiction handling | Were important conflicting findings and alternative interpretations addressed? |
| Qualification | Does language match evidence strength, applicability, and uncertainty? |
| Scope fidelity | Did the answer contract or expand beyond the governing context improperly? |
| Relevance | Is each major section materially useful to the question? |
| Synthesis | Does the response integrate evidence and explain relationships rather than concatenate summaries? |
| Clarity | Can a reader understand the conclusion, reasons, uncertainty, and limitations? |

If no material defect is detected, preserve the draft as final. If defects are material and correctable from existing evidence, create a revision with provenance to the draft and critique. If the defect is missing evidence, return to Gate C / search rather than hallucinating a repair.

## Canonical registry

VERA-DAR uses a canonical registry to preserve immutable provenance and the relationships required to reconstruct consequential research and reasoning decisions. Publicly documented record families are:

| Record family | Public purpose |
| --- | --- |
| Research and search | Governing questions, evidence requirements, search iterations, and candidate decisions. |
| Sources and provenance | Canonical source identity, acquired versions, retrieval history, and immutable provenance. |
| Evidence and appraisal | Exact evidence locations, critical-appraisal results, uncertainty, and integrity findings where applicable. |
| Evidence relationships and reasoning | Support, contradiction, qualification, derivation, claims, assumptions, alternatives, and inferential relationships. |
| Context decisions | Acquisition, admission, qualification, rejection, adequacy, and related decision rationale. |
| Answers and evaluations | Draft/final answers, consequential claims, evidence links, baseline responses, and quality evaluations. |

The implementation preserves lineage across these families so that consequential answer claims and evidence decisions can be traced backward to their supporting records without exposing the internal database schema in this public overview.

## Feedback and back-propagation

VERA-DAR preserves validated historical observations about search, appraisal, evidence decisions, reasoning, and answer quality. These derived records may inform subsequent prioritization and evaluation, but they never override current-run evidence, substitute for current uncertainty, or rewrite historical provenance.

## Agent and tool boundary

| Capability | Approach | Capstone responsibility |
| --- | --- | --- |
| Search | Reuse tool/API | Plan queries, discover candidates, register provenance, maintain search iterations. |
| Fetch/extract | Build (REQUIRED, DESIGNED, NOT IMPLEMENTED) | Full-document acquisition of each selected source into quarantine, with source versioning and acquisition status. |
| Deterministic integrity | Build small scripts | Inspect synthetic poison/injection patterns and exact spans. |
| Segmentation | Simple implementation/library | Create evidence units with exact source spans. |
| Critical appraisal | Build bounded evaluators/rubrics | Evaluate source process, data/evidence quality, applicability, uncertainty, lineage, and directness. |
| Lateral verification | Reuse search + bounded policy | Verify consequential source claims using external evidence. |
| Meta-drift | Build bounded evaluator | Store scope displacement and relevance separately. |
| Bias/epistemic | Build | Apply reference contract and explicit charge/weight/uncertainty. |
| Argument construction | Build | Link claims, support, refutation, assumptions, alternatives, and qualifiers. |
| Adequacy Gate C | Build | Decide whether the evidence set is sufficient or search must continue. |
| Registry | Build | Canonical provenance, decisions, findings, reasoning links, feedback, and memory. |
| Context construction | Build | Create inspectable reasoning context rather than unstructured retrieval bundle. |
| Model inference | Reuse frontier model | Generate from engineered context under a fixed synthesis contract. |
| Answer critique/revision | Build bounded evaluator + one revision | Measure and remediate material answer defects. |
| Evaluation | Build cases/harness | Measure retrieval, appraisal, grounding, provenance, inference, adequacy, drift, answer quality, and comparative lift. |

## Course session mapping

| Course layer | Increment | Definition of done |
| --- | --- | --- |
| Pre-course | FastAPI skeleton, `/health`, GitHub, environment | Service runs locally and project is ready. |
| Session 1 — Foundations | Reliable `POST /ask`, structured outputs, question interpretation, Evidence Requirements Map, explicit statuses | Question, evidence requirements, and bounded candidate assessment return validated JSON. |
| Session 2 — RAG | Search/retrieval, selective acquisition, provenance, citations, evidence lineage | System distinguishes candidate, acquired, admitted, and reasoning context and can trace evidence to exact source spans. |
| Session 3 — Agents | Search, fetch, registry, integrity, appraisal, lateral verification, Gate B, Gate C tools | Agent autonomously completes the bounded inquiry loop and can search again when evidence is inadequate. |
| Session 4 — TRACE evals | Curated cases for retrieval, integrity, provenance, appraisal, bias refusal, drift, inference, grounding, answer quality, comparative lift | Baseline and improved policies are compared with observable dimension-level metrics. |
| Session 5 — Memory | Persistent source, lineage, decision, argument, answer-evaluation, and derived-stat records | Prior validated history informs later runs without overriding current evidence. |
| Production / Demo | Deployed service + compact UI + baseline comparison | End-to-end workflow and measurable quality lift are understandable in about three minutes. |

## `POST /ask` contract

The public service contract exposes the research answer together with enough evidence, decision, citation, evaluation, and status information to make the result inspectable without exposing VERA-DAR's complete internal record topology.

```json
{
  "answer": "...",
  "citations": [],
  "evidence": [],
  "decisions": [],
  "evaluation": {},
  "status": "supported"
}
```

Detailed provenance, uncertainty, reasoning, and evaluation records remain available to the system's audit layer. A generic scalar confidence field is not the governing contract; dimension-level quality and uncertainty remain inspectable.

**Current implementation (2026-10-14): abstract-grounded; no full-document acquisition.** The contract above is the target. Today `/ask` writes answers from retrieved index records and their abstracts or search snippets and checks each cited claim against the cited abstract or snippet (full-document acquisition: REQUIRED, DESIGNED, NOT IMPLEMENTED; DEC-FDA1, DEC-OD10). Each returned source carries source-record provenance only: provider, identifier, retrieval time and the licence gate's decision on the metadata licence VERA-DAR records for the provider (not verified at source). It carries no immutable document version, no content hash, no passage location and no explicit inspection-scope label. The label is a recorded requirement (see "Evidentiary-inspection ceiling").

## TRACE-oriented and comparative evaluation

| Test family | Failure example | Observable |
| --- | --- | --- |
| Evidence requirements | Material aspect of question never represented in research plan | Requirement coverage and missed-requirement rate |
| Candidate selection | High-value candidate discarded before fetch | Recall/precision on curated candidate sets; Gate A false-negative rate |
| Integrity | Synthetic poison/injection span missed or benign text falsely flagged | Detection precision/recall on synthetic corpus |
| Source/process appraisal | Source reputation substituted for methodological evidence | Rubric agreement and unsupported-appraisal refusal |
| Lineage/independence | Ten derivative reports counted as ten independent confirmations | Lineage correctness / independence-group agreement |
| Grounding | Answer contains consequential claims unsupported by admitted evidence | Claim-to-evidence support completeness |
| Provenance | Evidence cannot be traced to source version and exact span | Trace completeness |
| Bias contract | Bias declared without target/reference/criterion | Unsupported-bias refusal rate |
| Meta-drift | Relevant expansion treated as irrelevant or drift admitted as direct evidence | Coordinate/classification agreement |
| Gate C adequacy | System synthesizes despite missing material evidence requirement | Adequacy decision agreement; search-again correctness |
| Inference | Conclusion does not follow from evidence or ignores material alternative | Argument/warrant evaluation agreement |
| Answer quality | Capstone answer is inferior to direct frontier-model baseline | Dimension-level blinded preference/lift |
| Revision | Critique worsens answer or hides unresolved evidence gap | Before/after dimension changes and regression rate |
| Memory | Historical summary silently overrides contradictory current evidence | Memory precedence correctness |

## Comparative frontier-model evaluation

The capstone isolates the value of context engineering by holding the research question and, where feasible, the final synthesis model constant. A strong design compares a direct response `M(Q)` with the same model operating over engineered context `M(Q, C*)`. Additional leading models may be included as external direct baselines.

For each evaluation question `i`: `ΔQ_i = Quality(R_capstone,i) - Quality(R_direct,i)`. Preserve the distribution of dimension-level differences rather than relying only on one aggregate score.

| Quality dimension | Minimum comparison |
| --- | --- |
| Correctness | Are factual and inferential conclusions accurate relative to the available evidence? |
| Completeness | Does the answer address all material aspects of the governing question? |
| Relevance | Does the response focus on information that materially serves the question? |
| Grounding | Are consequential claims supported by traceable evidence? |
| Synthesis | Does it integrate and reconcile evidence rather than list summaries? |
| Counterevidence | Does it surface and appropriately resolve or retain material contradiction? |
| Uncertainty | Does qualification match evidence strength and unresolved missingness? |
| Clarity | Is the reasoning understandable without sacrificing material nuance? |

"Significantly superior" is a measured outcome target, not an architectural assumption. One Demo Day example can demonstrate capability; repeatable superiority requires a multi-question evaluation set and a predefined judging method.

## Demo Day script

- Enter one bounded research question and show its Evidence Requirements Map.
- Show candidate sources and Gate A priority decisions, including at least one uncertain candidate retained to protect recall.
- Fetch selected candidates into quarantine and register hashes/provenance.
- Run deterministic integrity checks, including one synthetic offending example.
- Show critical appraisal and at least one lateral verification or evidence-lineage relationship.
- Show Gate B evidence decisions and the resulting support/refute/qualify relationships.
- Show Gate C evidence-set adequacy. Ideally demonstrate one unmet requirement triggering a targeted second search iteration.
- Construct the reasoning context and generate the draft answer.
- Show critical answer evaluation and one bounded revision if materially warranted.
- Present the final grounded response beside direct responses from the selected leading-model baselines.
- Open the audit view and trace one consequential answer claim backward through argument, evidence, exact source span, source version, and method.

The observer should be able to answer four questions: What evidence did the agent seek and choose? Why did it choose or reject it? How did the evidence warrant the final conclusions? Did the engineered system produce a materially better answer than the direct-model baselines?

## MVP, stretch, and deferred scope

The full design is the architectural envelope. **Addendum A is authoritative for the October 12 MVP implementation boundary.** The MVP retains only the capabilities required to test whether engineered evidence context improves final response quality or is necessary to make that improvement auditable.

Capabilities such as richer bias analysis, full meta-drift implementation, generalized evidence-independence graphs, broader integrity detection, extensive feedback analytics, multimodal ingestion, and distributed orchestration remain stretch or deferred work unless empirical testing demonstrates that one is necessary to the MVP value test.

## Acceptance criteria

- One capstone one-liner describes the implemented text evidence-and-reasoning system.
- `GET /health` and `POST /ask` operate through FastAPI with validated structured outputs.
- Each research question produces explicit evidence requirements before retrieval.
- Search candidates are prioritized before acquisition, with uncertainty preserved and Gate A tuned to avoid premature exclusion of plausible high-value evidence.
- The complete accessible document of each selected source is staged, hashed, and registered before semantic admission; failed or blocked acquisition is recorded, never replaced by an abstract. (REQUIRED, DESIGNED, NOT IMPLEMENTED for Demo Day: an accepted schedule/scope divergence (DEC-OD10), not a waiver or retirement of DEC-FDA1; see "Current state vs required")
- At least one deterministic content-integrity tool is invoked by the agent.
- Semantic evidence is traceable to exact source version and span.
- Source/process appraisal, data/evidence quality, applicability, independence, uncertainty, bias, and integrity remain separable canonical signals.
- At least one consequential source can be laterally verified or traced to an underlying primary/derivative evidence lineage.
- Scope displacement and contextual relevance remain separate measurements.
- Bias is not reported without an explicit reference-relative contract.
- Gate B context decisions are auditable and include at least admit, qualify, and reject; quarantine/excision is demonstrated with synthetic content if implemented.
- Gate C evaluates evidence-set adequacy and can trigger a targeted search-again iteration.
- Material conclusions link through explicit argument relations to supporting and contradicting evidence, assumptions, qualifiers, and alternatives where applicable.
- The first generated answer is evaluated on predefined critical-quality dimensions; one bounded revision occurs only when warranted.
- Every consequential answer claim can be traced to evidence, source version, provenance, and evaluation method.
- Comparative evaluation includes direct responses from at least one frontier baseline and measures dimension-level response-quality lift.
- The capstone is not considered successful merely because it is grounded and auditable; the final answer must demonstrate material incremental quality relative to the baseline for the Demo Day question.
- Persistent records support cross-session memory and derived feedback without replacing current-run evidence.
- The live system and UI demonstrate the principal workflow in approximately three minutes.

## Governing invariants

- **Evidence invariant:** One semantic datum, one target, one proposition or criterion, one context, one method, and one temporal scope per atomic assessment. A bias assessment additionally requires one explicit reference.
- **Context invariant:** No acquired content becomes model context merely because it was retrieved. Retrieval creates candidates. Inspection and appraisal create evidence. Argument construction creates reasoning structure. Policy creates context.
- **Adequacy invariant:** The existence of admissible evidence does not imply that the evidence set is sufficient. Final synthesis requires demonstrated coverage of material evidence requirements or an explicit qualified or insufficient-evidence status.
- **Reasoning invariant:** No consequential conclusion is warranted merely because supporting evidence was admitted. The conclusion must preserve an explicit inferential path, address material counterevidence and alternatives where present, and retain uncertainty arising from evidence, method, inference, and missingness.
- **Provenance invariant:** Original acquired content is immutable. Excision, transformation, aggregation, appraisal, validation, argument construction, answer critique, and revision create derived records linked to their parents.
- **Feedback invariant:** Historical source, method, and evaluator statistics may inform later prioritization but never substitute for current-run evidence, coverage, method, or uncertainty.
- **Resource invariant:** Every consumed resource — model/API calls, tokens, compute, wall-clock latency, and cross-instance or cross-process state — is finite and propagates or settles at a bounded, nonzero rate; no component may assume free, instant, or unbounded access to any of them. No component may consume a costed or rate-limited resource beyond what is strictly necessary to produce its required result, and no control's correctness may silently depend on unbounded consumption or instantaneous shared state across instances. Where bounded consumption or true cross-instance consistency cannot yet be guaranteed, the limitation is stated explicitly rather than presented as resolved.
- **Value invariant:** Grounding and auditability are necessary constraints, not substitutes for response quality. If the engineered pipeline adds material cost and complexity but produces a response no better than the direct leading-model baseline, it has not demonstrated sufficient incremental value for that task.
- **Source-custody invariant (2026-10-01):** No raw external source documents are kept in the repository. Acquired sources are stored in the database as immutable `sources` rows and nowhere else.
- **Licence invariant (R1 rule of 2026-10-03, applied in code 2026-10-04):** Sources are gated by licence before use. Reject only ND, no-educational-use and strictly-for-fee terms; plain NC is accepted; an undeclared or unclear licence is held for human review (HitL), never admitted by default. The original source's licence governs, and only if the original source is used. News pages are allowed when every use is cited in-line and source-referenced. (Supersedes the 2026-10-01 text, which precluded any NC licence.) Current status, measured figures and accepted residuals: README section "Status and limitations (2026-10-04)".
- **Quotation invariant (2026-10-01):** Outputs, including demo output, carry only targeted, relevant short quotes, never significant portions of source text, each with a source reference.
- **Authority invariant:** Retrieved or acquired content is evidence, not instruction. No text obtained through search, fetch, or evidence retrieval may directly authorize a tool call, filesystem change, network action, secret disclosure, memory write, configuration change, or message send; authority remains with deterministic policy and explicit tool schemas, never with retrieved data.

## Public disclosure boundary

This public overview documents VERA-DAR's evidence acquisition, critical appraisal, provenance, evidence-selection, reasoning-context, synthesis, and evaluation architecture at the level needed to understand and evaluate the capstone.

## Research references

1. Facione, P. A. (2023 update). *Critical Thinking: What It Is and Why It Counts.* Insight Assessment. <https://insightassessment.com/iaresource/critical-thinking-what-it-is-and-why-it-counts/>
2. Association of College and Research Libraries. (2016). *Framework for Information Literacy for Higher Education.* <https://www.ala.org/acrl/standards/ilframework>
3. Lefebvre, C., Glanville, J., Briscoe, S., et al. (2025). *Chapter 4: Searching for and selecting studies.* Cochrane Handbook for Systematic Reviews of Interventions, version 6.5.1. <https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-04>
4. Wang, R. Y., & Strong, D. M. (1996). *Beyond Accuracy: What Data Quality Means to Data Consumers.* Journal of Management Information Systems, 12(4), 5–33. <https://doi.org/10.1080/07421222.1996.11518099>
5. Critical Appraisal Skills Programme. (2026). *What is critical appraisal?* <https://casp-uk.net/what-is-critical-appraisal/>
6. Cochrane/GRADE. *Chapter 14: Completing Summary of Findings tables and grading the certainty of the evidence;* GRADE Working Group materials. <https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-14>
7. Wineburg, S., & McGrew, S. (2019). *Lateral Reading and the Nature of Expertise: Reading Less and Learning More When Evaluating Digital Information.* Teachers College Record, 121(11). <https://doi.org/10.1177/016146811912101102>
8. Sperber, D., Clément, F., Heintz, C., Mascaro, O., Mercier, H., Origgi, G., & Wilson, D. (2010). *Epistemic Vigilance.* Mind & Language, 25(4), 359–393. <https://doi.org/10.1111/j.1468-0017.2010.01394.x>
9. PRISMA. (2020). *PRISMA 2020 flow diagram and checklist.* <https://www.prisma-statement.org/prisma-2020-flow-diagram>
10. Facione, P. A., & Facione, N. C. *Holistic Critical Thinking Scoring Rubric.* Insight Assessment. <https://insightassessment.com/iaresource/the-holistic-critical-thinking-scoring-rubric/>

> **Transfer note:** VERA-DAR imports transferable principles from systematic-review, evidence-based practice, information-literacy, argumentation, and data-quality research. It does not claim PRISMA, GRADE, CASP, or Cochrane compliance for general web research.

---

# Addendum A — October 12 MVP and delivery milestones

## Purpose and governing constraint

This addendum converts the full architecture into the minimum vertical slice required to test the capstone thesis before **12 October 2026**. The full design remains the governing envelope. The MVP is not a miniature implementation of every capability in the full design. It is the smallest end-to-end system capable of testing whether deliberate, auditable evidence acquisition, critical evaluation, and context construction can cause a frontier model to produce a materially better research response than the same model produces directly.

**MVP proposition:** For a bounded research question, an engineered evidence context can produce a measurably better grounded response than a direct leading-LLM baseline, while preserving an inspectable chain from consequential evidence decisions to source, evidence, provenance, and evaluation method.

**MVP decision rule:** A capability is on the October 12 critical path only if it materially increases the probability of improving final response quality or is necessary to make that improvement auditable.

## A.1 Minimum end-to-end workflow

```
Question → Evidence Requirements → Search → Gate A → Fetch/Provenance → Evidence Appraisal
        → Gate B → Gate C → Context Construction → Draft Answer → Critique/Revision
        → Final Answer → Baseline Comparison
```

The three decision gates are deliberately simplified for the MVP:

| Gate | MVP question | Minimum outputs |
| --- | --- | --- |
| Gate A — acquisition | Is this candidate sufficiently plausible and useful to justify acquisition? | `fetch` / `defer` / `reject` |
| Gate B — admission | After acquisition and appraisal, may this evidence inform reasoning context? | `admit` / `qualify` / `reject` |
| Gate C — adequacy | Does the admitted evidence set adequately cover the material requirements of the research question? | `adequate` / `search_again` / `insufficient` |

## A.2 MVP required capabilities

| Capability | MVP definition of done |
| --- | --- |
| Service contract | `GET /health` and validated `POST /ask` through FastAPI. |
| Evidence requirements | Create a small, explicit map of the material findings or subquestions needed to answer the bounded research question. |
| Candidate discovery | Search the web and register candidate title, URL, snippet/rank, query/run, and Gate A decision. |
| Full-document acquisition | For each source selected by Gate A, acquire the complete accessible document, stage it, compute a content hash, and preserve retrieval provenance, version and acquisition status; record failures (see "Full-document acquisition"). (REQUIRED, DESIGNED, NOT IMPLEMENTED for Demo Day: an accepted schedule/scope divergence (DEC-OD10), not a waiver or retirement of DEC-FDA1; see "Current state vs required") |
| Semantic evidence | Extract machine-addressable evidence spans tied to the exact source version. |
| Critical appraisal | Evaluate each consequential evidence unit with a fixed, inspectable rubric. Minimum dimensions: relevance, source/process quality, claim support, applicability, uncertainty, and support/challenge relation. |
| Context decisions | Apply Gate B and retain the decision, rationale, contributing evidence, method/rubric version, and provenance. |
| Evidence-set adequacy | Apply a bounded Gate C coverage check. Permit at most one search-again iteration in the first MVP. |
| Reasoning context | Organize admitted/qualified evidence into a compact context that exposes support, contradiction, uncertainty, and source lineage without dumping the registry into the model prompt. |
| Grounded synthesis | Generate a cited draft answer using only the permitted reasoning context. |
| Answer critique | Evaluate the draft for correctness, completeness, grounding, synthesis, relevance, clarity, and material uncertainty. Permit one bounded revision when a material defect is detected. |
| Baseline comparison | Ask the same final model the same research question directly, without the engineered evidence context, and preserve that response as the primary control. |
| Comparative evaluation | Evaluate direct and engineered responses with a predefined rubric and preserve dimension-level results and evaluator rationale. |
| Audit view | Expose enough of the trace that an observer can determine what evidence was selected, what was rejected or qualified, why, and where each consequential decision originated. |

## A.3 Minimum canonical data model

The MVP uses the same public record families described above, implemented only to the depth necessary for the October 12 value test. At minimum it must preserve:

- the governing research question and evidence requirements;
- candidate discovery and acquisition decisions;
- immutable source/version provenance;
- exact evidence locations and critical-appraisal results;
- context/admission and evidence-set adequacy decisions;
- direct-baseline and engineered answers;
- consequential claim-to-evidence traceability; and
- dimension-level comparative evaluation results.

The implementation may use a compact schema, but it must preserve provenance, uncertainty, missingness, and immutable source lineage.

## A.4 MVP evaluation contract

The MVP must evaluate outcome quality as well as process integrity. Grounding and auditability do not compensate for an inferior final response.

| Dimension | MVP question |
| --- | --- |
| Correctness | Are consequential factual and analytical claims accurate relative to the available evidence? |
| Completeness | Does the response address every material part of the bounded research question? |
| Grounding | Are consequential claims supported by admitted or qualified evidence? |
| Synthesis | Does the response integrate evidence and reconcile material relationships rather than concatenate summaries? |
| Relevance | Does the response remain useful and responsive to the governing question? |
| Clarity | Is the reasoning and resulting conclusion understandable and appropriately organized? |
| Uncertainty | Are material evidentiary limitations, contradictions, and unresolved uncertainty represented appropriately? |
| Trace completeness | Can consequential evidence and answer decisions be traced to source/version/span, provenance, and evaluation method? |

**Primary comparative design:** where feasible, hold the final generator constant. Compare `M(Q)`, the model answering the research question directly, against `M(Q, C*)`, the same model answering with the MVP-selected reasoning context. This isolates the incremental value of the context-engineering pipeline more cleanly than comparing different generators.

## A.5 MVP acceptance criteria

1. One bounded research question completes the full MVP workflow without manual database intervention.
2. At least one candidate is evaluated before acquisition and every acquired source is versioned and provenance-linked.
3. Every consequential evidence unit used in the final answer is traceable to an exact source version and span.
4. Gate B demonstrates at least admit and qualify/reject actions with inspectable rationale.
5. Gate C explicitly returns `adequate`, `search_again`, or `insufficient`; at least one curated test demonstrates the search-again or insufficient path.
6. The final model receives a compact engineered reasoning context rather than an undifferentiated registry dump.
7. The first generated response is evaluated as a draft; one bounded revision is supported when material deficiencies are found.
8. The same research question is run as a direct leading-model baseline and retained for comparison.
9. Direct and engineered responses are compared on predefined quality dimensions rather than only an opaque scalar score.
10. The Demo Day case shows a materially superior engineered response. If it does not, the MVP records the result as a failed value test rather than redefining success.
11. An observer can answer: What evidence was chosen? What was rejected or qualified? Why? Can each consequential decision be traced to its source, evidence, provenance, and evaluation method?
12. The complete demonstration remains understandable in approximately three minutes.

## A.6 Explicitly deferred from the October 12 MVP

The following remain part of the full design but are not required to establish the first value test:

- Generalized or exhaustive bias taxonomy beyond the minimum appraisal needed by the selected research domain.
- Full two-axis meta-drift implementation and distribution-preserving drift summaries.
- Multiple content-integrity detectors, generalized poison/trojan detection, autonomous remediation, and production malware analysis.
- Bounded excision workflow beyond a synthetic demonstration, if any.
- Full lateral-verification automation and retraction/correction infrastructure.
- General evidence-independence graph beyond a simple shared-origin or derivation flag.
- Multiple evaluator-model agreement, adaptive evaluator calibration, and complex source-history scoring.
- Derived source dashboards, generalized feedback analytics, and broad cross-run reputation models.
- Multimodal ingestion, universal ontology architecture, adaptive semantic geometry, and distributed orchestration.

## A.7 Delivery milestones to 12 October 2026

Milestones are outcome gates rather than promises to implement every subsystem in the full design. A milestone is complete only when its artifact works end to end at the stated boundary.

| Milestone | Target | Deliverable | Exit criterion | Status (18 Sep 2026) |
| --- | --- | --- | --- | --- |
| M0 — Scope freeze | 9 Sep | Freeze the full design as the envelope and this addendum as the MVP boundary. Select the first bounded research question and baseline model. | No new feature enters MVP unless it improves response quality or auditability. | Done |
| M1 — Service and records | 10–13 Sep | Working `/health` and `/ask`; validated schemas; minimal persistent records for question/run/candidate/source. | One request persists and returns structured output with stable IDs. | Partial — service and schemas work; no persistence layer exists at all |
| M2 — Search and provenance | 14–19 Sep | Search, Gate A, selective fetch, staging/hash, source/version provenance. | A question produces registered candidates and acquired immutable source versions. | Missed — window closed with none of this built; see A.11 |
| M3 — Evidence and Gate B | 20–25 Sep | Evidence extraction, fixed critical-appraisal rubric, admit/qualify/reject decisions. | Every admitted/qualified evidence span has a source/version/span, assessment, rationale, method, and provenance. | Not started — blocked on M2 |
| M4 — Gate C and context | 26 Sep–1 Oct | Evidence-requirements coverage, bounded adequacy check, one search-again loop, compact reasoning-context construction. | System either constructs adequate context, searches once for a named gap, or returns insufficient evidence. | Not started |
| M5 — Synthesis and repair | 2–5 Oct | Grounded draft generation, claim-to-evidence links, answer critique, one bounded revision. | Final answer is cited, claim-traceable, and passes the MVP answer-quality checks or is explicitly qualified. | Not started |
| M6 — Baseline and eval | 6–8 Oct | Direct same-model baseline plus comparative evaluation harness. Curate initial evaluation cases. | For the Demo question, both conditions are stored and scored on the predefined dimensions with rationale. | Not started |
| M7 — Demo hardening | 9–10 Oct | Compact UI/audit view, deterministic demo path, latency/error handling, cached fallback artifacts where appropriate. | A three-minute rehearsal completes without hidden manual steps and exposes the evidence-decision trace. | Not started |
| M8 — Freeze and rehearse | 11–12 Oct | Code/content freeze except critical defects; repeat evaluation and Demo Day rehearsal. | Reproducible final run, preserved baseline comparison, known limitations documented, demo ready. | Not started |

Original targets above are preserved as originally written, per the Provenance invariant; see **A.11** for the recorded slippage and near-term replan.

## A.8 Stretch milestones after MVP stability

| Priority | Stretch milestone | Trigger |
| --- | --- | --- |
| S1 | Add bounded lateral source verification for consequential sources. | Only after M5 is stable and source credibility is a demonstrated quality bottleneck. |
| S2 | Add simple evidence-lineage/independence grouping beyond shared-origin flags. | When apparent corroboration materially affects conclusions. |
| S3 | Restore two-axis meta-drift at candidate, evidence, and answer stages. | When scope/relevance errors appear in evaluation cases. |
| S4 | Add one deterministic prompt-injection/content-integrity detector and synthetic test. | When Session 3 / tooling requirements are otherwise satisfied. |
| S5 | Expand evaluation from the Demo question to a 10–20 question blinded or model-independent comparison set. | After the end-to-end pipeline is stable. |
| S6 | Add persistent derived source/method feedback for Session 5 memory. | After current-run evidence precedence and provenance are verified. |
| S7 | Optimize latency, token use, model routing, and acquisition cost. | After quality lift is demonstrated; optimization must not erase the measured advantage. |

## A.9 Stop/go rules

- If a proposed feature does not improve answer quality, evidence adequacy, or traceability, defer it.
- If Gate A reduces recall on high-value evidence, loosen Gate A before adding more scoring complexity.
- If Gate C repeatedly declares adequacy despite obvious missing evidence, repair evidence requirements and coverage logic before adding downstream sophistication.
- If the final response is worse than the direct baseline, inspect evidence coverage, context construction, and synthesis before adding provenance or bias features.
- If the same-model engineered-context condition does not show material lift on the Demo question by M6, narrow or change the research question only if the new question remains a legitimate instance of the declared capstone domain. Record the failed case rather than discard it.
- Once the MVP demonstrates material quality lift and complete traceability, freeze the critical path. Stretch work must not jeopardize the October 12 demonstrable system.

## A.10 Relationship to the full design

This addendum does not supersede the full design. It defines the implementation boundary for the capstone MVP. The full design continues to govern conceptual semantics, provenance invariants, evidence and reasoning distinctions, and post-MVP evolution. Where this addendum simplifies a construct, the simplification is an implementation deferral rather than a conceptual redefinition.

## A.11 Status and replan (18 September 2026)

Actual status against A.7 is recorded in that table's Status column rather than by altering the original targets, per the Provenance invariant. Summary:

- **M0** — done.
- **M1** — partial. `GET /health` and validated `POST /ask` run; the "minimal persistent records for question/run/candidate/source" exit criterion is unmet — the running service has no persistence layer of any kind.
- **M2** — missed. No search, Gate A, selective fetch, staging/hashing, or source/version provenance exists. The window closed 19 Sep with none of this built.
- **M3 onward** — not started, and cannot meaningfully start until M2's foundation exists.

**Infra blocker underneath M1 and M2.** `main.py` already documents why model-pricing configuration is version-controlled JSON rather than a runtime database: Render's filesystem is ephemeral and does not survive a redeploy. The same constraint applies to every record M1–M3 need to persist. Closing M1's persistence gap and M2's provenance requirement therefore requires provisioning a real hosted database (e.g. Render managed Postgres) before any Gate A/B code can log a decision. This is an external provisioning step with its own lead time and has not been started — it should be treated as the first concrete action, ahead of writing Gate A/B logic itself.

**Governing rule invoked.** Per the A.9 stop/go rule — "if a proposed feature does not improve answer quality, evidence adequacy, or traceability, defer it" — the retrieval-quality digest's fuller scope (retrieval history, hybrid BM25/RRF, reranking) is deferred past M2/M3. Whenever it is picked up, it should extend rather than duplicate the gates above; implementing it now would build refinement onto a retrieval/evidence layer that does not yet exist.

**Near-term catch-up target.** Provision hosted persistence and complete the M2 slice (search, Gate A, selective fetch, provenance registration) by **23 September 2026** — a 4-day compression of M2's original 6-day window, reflecting the 4 days already elapsed since M2's original start with no output. Remaining milestones (M3–M8) should shift by the same delta rather than being individually re-dated here; assigning precise dates to work not yet scoped against real M2 output would be false precision. If 12 October is a fixed external date and this compression proves insufficient once M2 actually lands, the correct response per A.9 is to cut scope — defer S1–S7 stretch items further, or narrow M4's adequacy loop — before extending further into the M7/M8 freeze-and-rehearse window.

**Database host is course-scoped, not the long-term commitment (noted 2026-09-18, no code change).** The Render-managed Postgres instance referenced above is the near-term persistence layer for this course's remaining timeline. VERA-DAR's actual longer-term production home is expected to move to a non-Render installation eventually — but PostgreSQL itself is the durable decision, independent of host: whatever infrastructure VERA-DAR eventually runs on, the persistence layer stays Postgres, not a change in database technology. `vera/db.py`'s `INTERNAL_DB_URL`/`EXTERNAL_DB_URL` split is a Render-specific mechanism (keying off Render's own private-network URL distinction) that will need to be revisited at that migration — not because Postgres changes, but because the internal/external URL split is specific to Render's own networking model. No action needed now; recorded so a future session doesn't mistake the current Render wiring for a permanent architectural commitment to Render itself.

### A.11 addendum: status

Status: see the root README.

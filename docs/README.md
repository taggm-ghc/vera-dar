# VERA-DAR documentation index

*Rewritten 2026-10-01 (R7a author, R11a record role). This file previously duplicated the root
README; that copy is archived at
an internal archive copy (not published). The project overview,
status, run instructions and environment description are in the root [README.md](../README.md).*

## Documents in this folder

Status tags: LIVE (current, maintained), RECORD (kept as written, dated annotations only), SUPERSEDED, BACKLOG (parked, not built). Listing verified against the directory 2026-10-02. Files named `*-prv.*` are private and gitignored (`.gitignore`: `*-prv.*`), so they exist locally only.

| Document | Purpose | Status |
|---|---|---|
| [vera-design.md](vera-design.md) | Full design, governing invariants, Addendum A (MVP boundary, milestone table, A.11 status and replan, status addendum). Original milestone targets preserved per the Provenance invariant. | LIVE |
| VERA-OCT-15-SPRINT-PLAN.md (not published) | Oct 15 - Nov 1 sprint plan (2026-09-30). Fallback plan, superseded in part by the 2026-10-03 MVP target. | RETIRED (2026-10-12; left unedited) |
| pre-session-checkin.md (not published) | Capstone seed and pre-session check-in (8 September 2026). | RECORD |
| vera-v06-design-prv.md (not published) | Earlier private design, Refined Version 06 (9 September 2026); vera-design.md is current. | RECORD |
| vera-retrieval-quality-digest-prv.md (not published) | Retrieval-quality layer digest (18 September); nothing implemented, deferred by A.9. [Corrected 2026-10-04: digest layer still unbuilt, but a search-level retrieval build (item #71) exists, uncommitted; see the digest's 2026-10-04 annotation.] | BACKLOG |
| vera-retrieval-vocabulary-reconciliation-prv.md (not published) | Maps the digest's terms onto Gate A/B/C; still cited by vera-design.md A.9 (verified 2026-10-02). | LIVE (vocabulary only) |
| vera-multimodal-extraction-backlog-prv.md (not published) | Parked OCR/table/figure extraction work; scope decision pending. | BACKLOG |
| vera-privileged-mediation-annotated-bibliography-prv.md (not published) | Annotated bibliography for the privileged-mediation security framework (September 2026). Not reviewed in this pass; the file is root-owned and read-only to the maintainer, so it carries no annotation. | RECORD |
| `vera-eigenprojection-distribution-retrieval-design-prv.docx` | Word design note (24 September); not Markdown, not reviewed here. | RECORD (UNVERIFIED content) |

The design-notes files at the repo root (`../ask-guardrail-design-prv.md`, `../pricing-scraper-design-prv.md`) and the folders `../opencode.cli/` and `../refactor-prv/` are RECORD material; each has its own README or dated annotations.

Database documentation: [../db/README.md](../db/README.md). Environment variables and accounts, described
without secrets: [../.env.example](../.env.example).

## Planning records (private, not published)

Status, plan, flows, state report and the longitudinal evaluation design are listed in the
"Documentation Index" section of the root [README.md](../README.md), together with the dated history
archive folders (`2026-10-01-cleanup-archive` through `-archive-10`).

*Annotation 2026-10-02 (R11a, round 3): index completed (it previously omitted the private files) and status tags added. Pre-edit copy: an internal archive copy (not published). The pipeline cannot yet run end to end; live plans are in the owner's private planning notes.*

*Annotation 2026-10-04 (R11a orchestrator): links verified against the directory. The bibliography file is root-owned and read-only; it was not annotated. Current status of the pipeline is in `vera-design.md` (2026-10-04 annotation).*

*Annotation 2026-10-12: status as of 7fc7fe1 (pushed, deploy unverified) is in the root [README.md](../README.md) and the 2026-10-12 annotation at the end of `vera-design.md`. `VERA-OCT-15-SPRINT-PLAN.md` (not published) is retired and left unedited.*

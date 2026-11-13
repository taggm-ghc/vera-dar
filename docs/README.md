# VERA-DAR documentation index

[← Project overview](../README.md)

Current project status (dated) is kept in one place: [Status](../README.md#status) in the project overview. Documents here are for developers; end-user explanation is in the project overview and the About page. Rules for coding and research agents working in this repository are in [AGENTS.md](../AGENTS.md).

## Overview documents

| Document | Contents |
|---|---|
| [API and data use](api.md) | Public endpoints, response semantics, privacy and provenance limits |
| [Architecture](architecture.md) | Public conceptual architecture and known gaps |
| [Development](development.md) | Local requirements, startup and testing |
| [Deployment](deployment.md) | Docker and Render (interim host) setup for separate API and UI services |
| [Evaluation and findings](evaluation.md) | TRACE methodology, historical results (build 0094955) and limitations |
| [Citations, sub-queries and storage terms](citations.md) | APA 7 house variant, sub-query decomposition, search lanes and page-access gates |

## Design and checks

| Document | Contents |
|---|---|
| [vera-design.md](vera-design.md) | Full design, governing invariants, dated current-state blocks (a dated Status block of 2026-11-13 lists superseded sections) and historical milestone plan |
| [release-checks.md](release-checks.md) | Release gate check procedures |
| [lanes-enable-checklist.md](lanes-enable-checklist.md) | Checks before the Official or Current-reporting lane is enabled |
| [trace_eval/](trace_eval/) | Frozen TRACE evaluation label files (`check_labels_v1.json`, `support_labels_v1.json`; historical) |
| [../db/README.md](../db/README.md) | Database schema, accounts and migrations |
| [../config/wordlists/README.md](../config/wordlists/README.md) | English word list used by spell-clarify: source, licence, build |

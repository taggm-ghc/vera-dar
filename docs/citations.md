# Citations, sub-queries and storage terms

[← Project overview](../README.md)

Status of everything on this page: **offline-tested only; live behaviour unverified.** (Updated 2026-11-05: the Current-reporting lane is enabled in config at HEAD, see "Flag state"; nothing here has been validated live.) It describes code from the second NONASK-1103 push. The search-lane section and the page-access section describe code prepared for that push. The lanes apply the separate read, use-for-answer and ingest gates (see "Page-access gates").

## APA 7 house variant: identifiers appended

Sources are cited in APA 7 with one house rule: an identifier (a link or ID) is appended to every reference, so a reader can reach the source. The caption shown under a reference list is "APA 7 (VERA-DAR house variant: identifiers appended)". Where the source's licence is known (from its attribution record: Europe PMC and arXiv full text), the reference also ends with "Licence: <name>" linked to the licence URL (plain text: "Licence: <name> (<URL>)"); when no licence is known nothing is added. A flag in `config/citation_style.json` selects `apa7` (default) or `legacy`; an unreadable or unknown value means `apa7`, and `legacy` returns no formatted strings.

Principles: metadata is never invented. A missing author moves the title to the author position; a missing date prints "n.d."; a joined author string is kept as one literal author, never split. An invalid identifier is left out, never repaired. If the formatter fails, the answer path is unaffected and the citation fields are empty.

### Identifier precedence

1. If the text actually read was the arXiv version, the arXiv ID comes first, followed by the DOI when there is one.
2. Web pages, news, datasets and software use their URL.
3. Otherwise the first available of: DOI, Handle, arXiv ID, PMCID, PMID, ISBN (books and chapters only), then the URL.
4. At most one secondary identifier is added, and only when it reaches a different resource (DOI plus PMCID, or DOI plus ISBN).

A PMCID appears in a reference only when no DOI or arXiv ID is available.

### Validation

Each identifier must pass a format check: DOI (bare or `doi.org` form, no spaces or quote or bracket characters), arXiv (new `YYMM.NNNNN` or old `archive/NNNNNNN` form, optional version), PMCID (`PMC` plus digits), PMID (up to 8 digits), Handle (a `hdl.handle.net` https link or an explicit field), ISBN (ISBN-10 or ISBN-13 with a valid check digit). Links are shown only for public hosts; local or private addresses are never linked. Identifiers that came from a different provider record are not attached.

### Where references appear

- the answer area: in-text citations such as "(Alpha, 2020) [1]" replace bare numbers, and a References block follows the answer;
- the reading list (candidate documents);
- the Report page.

References **never** appear in the sidebar, which stays free of per-source detail.

### Public fields

Each public source and candidate carries three new strings, and only these: `reference` (the formatted APA 7 entry), `in_text` (the parenthetical, with a/b suffixes where author and year clash) and `characterisation` (a short line: type, outlet or site, date, for example "journal article · Journal name · 2020"). Raw identifiers, author lists, access dates and source type stay out of the public record except the fields already public (for example `identifier`, `pmcid`). The evidence trace's public rows carry no URL or reference; the development view does.

## Per-sub-query adequacy in the public trace

When a question is decomposed into two or more sub-queries, the adequacy check judges each one by the same coverage rule as the whole question (no new threshold). The overall verdict is then: finalize when every sub-query finalizes, abstain when none is supported, otherwise qualify. A sub-query with no support is reported as not found rather than hidden. In the public trace, `adequacy.subqueries` is a list of `{label, verdict}` only (label at most 300 characters). With zero or one sub-query the record is unchanged and has no `subqueries` key.

Sources that are reported by a third party rather than verified carry the trace label "reported by the source, not independently verified" (`config/evidence_trace.json`, key `epistemic`), never "supported".

## Sub-query decomposition

Decomposition is deterministic (no model call, network or database). The original question stays authoritative and is kept unchanged; the parts are extra retrieval handles built only from the question's own words. Rules, in order: an entity list of two or more items (separated by comma, "and", "or", "&" or "/"); a comparison ("X vs Y", "compare X with Y", "difference between X and Y"); otherwise one part. At most 4 parts: any overflow is merged into the last part, nothing is dropped. A model-based fallback exists as a stub and ships off. `max_parts` (4) and the comparison side length (8 words) are uncalibrated priors in `config/subquery.json`; the unit tests are not a calibration.

## Storage and terms policy

In plain words (owner decisions R-037 and R-042):

- Some pages have no open licence. For the Official and Current-reporting lanes, such a page may be fetched while a question is being answered, read for that question only, described in VERA-DAR's own words, quoted in short excerpts (200 characters or fewer) and cited in APA 7.
- Such text is **never stored**: no full text, chunks or embeddings enter the corpus, and no page text goes into logs or traces. Only the bibliographic record, the link, the retrieval time and the licence observation may be kept. Storing anything still requires the licence gate.
- For the Official lane, a page is fetched only where both the site's terms and its robots.txt allow automated access. Otherwise the answer gives the link only. An allowed fetch is not a licence beyond the terms observed.
- In the public evidence trace, a page of this kind shows no excerpt.
- Questions' sub-query text is sent to the third-party search services used for current reporting (see the privacy section of the [README](../README.md#privacy-and-data-handling)).

## Search lanes

Status: offline-tested only; live behaviour unverified. As of 2026-11-05 the Official lane ships off and the Current-reporting lane is on in config (Superseded 2026-11-05: earlier text said both ship off). The checks that must pass before either is enabled are in the [lanes enable checklist](lanes-enable-checklist.md).

### Update 2026-11-06 (NEWS-BATCH-2)

The lane notes below predate these changes (offline-tested only). Wikipedia-citation links are shown link-only, credited "via Wikipedia citations", never read (NT-3). VOA: full text only for VOA-original pages (path pattern, "By VOA News" byline, no wire markers); Wikinews: `/wiki/` mainspace only, archive of a closed site; ProPublica and The Conversation: excluded; PBS: link-only; FEC and Census Bureau registry rows added, EAC suspended, Official lane flag still off (H5). The no-derivatives access module (H19) is not wired in and is link-only until NC-BASIS-1.

### Lane names and sources

- **Official lane.** For each entity named in a sub-query that has an active row in `config/entity_registry.json`, the lane uses that row's own documentation entry URLs. No host is guessed. An entity with no active row gives nothing. By default there is no search call (see `site_search` below).
- **Current-reporting lane.** *Superseded 2026-11-05: Hacker News is no longer in the fallback order (GDELT only).* For each sub-query that asks about the current state, the lane queries GDELT (a news index) and, as fallback, Hacker News search (Algolia), in that order. It sends extracted search terms, not the sentence. Each hit is a link with title, date and outlet. The portal name is not shown to the user.

### How a lane result is labelled

- A link is labelled "Official" or "Reported".
- A page that was read is labelled "read in full for this answer; not stored".
- In the answer, a lane claim is called "official" or "reported", never a verified fact. The text says "according to <vendor> documentation" or "as reported by <outlet>".
- Lane pages are used for that one answer only. They are never stored. Quotes are 200 characters or fewer.
- A lane page that fails the relevance check gives a link only.
- The public record has links, a count of pages read and a credit line (GDELT is credited with a link). It has no page text.
- When a lane is on, the data-use notice names GDELT and Hacker News search and says that pages read are not licence-cleared or stored (see the [README](../README.md#privacy-and-data-handling)).

### Which pages are read

- Both lanes use the shared permission record in `vera/source_permissions.py` (owner decisions R-049 and R-050). A page is read only when its read gate passes: robots.txt permits the fetch (checked by the shared fetch module on the same connection, failing closed), the registry `access_verdict` is not "forbid", the host is not on the owner deny-list (`config/source_permissions.json` `deny_hosts`, empty as shipped), and there is no login, paywall or no-crawl signal. Unknown means allowed.
- A read page is used in the answer only when its use-for-answer gate passes: no No-Derivatives licence, no no-AI signal (`noai` or `Content-Signal` `ai-input=no`) and no registry `derivative_verdict` "forbid". Otherwise it is a reading-list link only.
- As shipped, the registry marks AWS, Nvidia and Microsoft Azure documentation `access_verdict` "forbid" (never fetched, links only) and Google Cloud "allow"; the other seeded rows are "unknown".
- Current-reporting lane: outlet pages linked from GDELT or Hacker News pass the same gates, with at most `admission.max_spans_total` outlet pages read per request.

### Sub-queries and search

- The deterministic decomposer (see "Sub-query decomposition") is on in `config/subquery.json` (`enabled: true`). The lanes run once per sub-query.
- Per-sub-query academic search is a separate switch: `config/auto_acquire.json` `search.subquery.enabled` ships `false`. While it is false, the academic search is the single, unchanged search. When true, OpenAlex, Europe PMC and arXiv run once per part, and the per-job search-call budget grows from 7 to 7 + 3 x (parts - 1), at most 16 for 4 parts. It ships off because calibration is not done (see the note in that file). Live rate-limit behaviour and latency are untested.

### Flag state as shipped

| File | Key | Value |
| --- | --- | --- |
| `config/subquery.json` | `enabled` | true |
| `config/subquery.json` | `official_lane.enabled` | false |
| `config/subquery.json` | `official_lane.site_search` | false |
| `config/subquery.json` | `model_fallback.enabled` | false |
| `config/reporting_lane.json` | `enabled` | true at HEAD (Superseded 2026-11-05: was false) |
| `config/reporting_lane.json` | `fallback_order` | ["gdelt"] only (Hacker News no longer in the order) |
| `config/reporting_lane.json` | `outlets` | 18 listed hosts at HEAD (was empty) |
| `config/reporting_lane.json` | `emit_link_only` / `outlet_query` | true / "all" in the working tree (owner decision 2026-11-05; HEAD: false / "allowlist") |
| `config/auto_acquire.json` | `search.subquery.enabled` | false |
| `config/synthesis.json` (synthesis pass) | `enabled` | false |
| `config/research_loop.json` (research loop) | `enabled` | false |

With `site_search` false the Official lane uses the registry entry URLs only. When true, each "allow" entity gets one `site:<host>` query through the shared DuckDuckGo Lite provider, with its cache, daily cap and rate gate. The GDELT field names are not confirmed against a live response, and the Hacker News search terms are unconfirmed, so the Current-reporting lane was kept off at the time of writing; it is now on in config (see above), still without live confirmation.

The synthesis pass and the research loop are present but not wired into a request, and both ship off.

## Page-access gates

Owner decisions R-049 and R-050 (2026-11-03) set three separate gates, implemented in `vera/source_permissions.py`:

- **Read:** read unless restricted (robots.txt, registry `access_verdict` "forbid", owner deny-list, login, paywall or no-crawl signal). A restricted host is never fetched, not even for one answer; it appears only as a link.
- **Use for answer:** allowed unless barred (No-Derivatives licence, no-AI signal, registry `derivative_verdict` "forbid"). A barred page is a reading-list reference only.
- **Ingest:** only when the licence explicitly permits storage and the registry `ingest_verdict` is not "forbid". Lane pages are never stored, whatever this gate says (R-037).

In this code the search lanes use these gates. Their use on the general acquisition path is part of a later push-2 phase and is not described here.

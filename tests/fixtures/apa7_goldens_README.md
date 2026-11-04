# APA 7 goldens (NONASK-1103-C-GOLD)

`apa7_goldens.json` holds golden cases for the APA 7 formatter (house variant: identifiers appended). An
independent golden author wrote them from two sources only: the V-1 and V-1b evidence record
(`work/NONASK-1103/citation-v1-evidence.md` in the workspace root, not in this repository) and the identifier
rules in section 7 of `work/NONASK-1103/citation-plan.md`. The author did not read the formatter implementation.
All records are synthetic. Titles, names and identifiers are invented. Every URL uses `example.org`,
`example.com` or a fixed public resolver host.

## Case shape

- `input`: a CSL-JSON-shaped record. Authors come either as a structured list (`family`, `given`) or as one
  `literal` element, which is a verbatim joined string or a group name. The record also carries `issued`,
  `title`, `container-title`, `publisher`, `DOI`, `URL`, `ISBN`, `number`, `version`, `accessed` and `genre`,
  plus the VERA extensions `ids` {handle, arxiv, pmcid, pmid}, `vera_class`, `provider` and `fulltext_url`.
  `_same_record` marks whether an identifier comes from the same provider record as the title. The test adapter
  may map these keys onto the implementation's frozen key list (C1.1), but must not change their values.
- `kind`: omitted for a single record. `list` means `references_in_order` plus `in_text_by_title`. `answer`
  means `in_text(recs, ns)` substitution on an answer text. `property` describes a generator.
- `expected.reference` and `expected.in_text`: the visible text (see below). `links` lists the expected link
  targets in order; `[]` means no link at all. `reason_code` or `reason_code_prefix` names the logged reason code.
- `gate` applies to the reference string. `gate_in_text` applies to the in-text citation. `tags` give the
  evidence level (V2, U, CONFIRMED-PRIMARY, CONFIRMED-SECONDARY, HOUSE, HOSTILE). `evidence` points to the V-1 row,
  the V-1b row or the plan section behind the case.

## visible(raw)

Comparisons use the visible text, not the raw Markdown:
1. Replace each unescaped `[text](href)` with `text`, and record `href` as a link.
2. Remove a backslash that precedes ASCII punctuation (the `plain()` / `plain_url()` escapes).
3. Keep unescaped `*` emphasis markers, so `*Journal*` means italic.

`raw_*` assertions apply to the raw string before step 1.

## Gates

- `exact`: `visible(reference)` equals `expected.reference` after two normalisations. (a) `*` markers are
  removed from both strings, because the primary text came from pdftotext, which cannot show italics. Italic
  placement is still checked as structure. (b) In a page range, U+2013 and `-` are treated as equal, because the
  evidence quotes show both. A case is `exact` only when V-1b rates that format CONFIRMED-PRIMARY or
  CONFIRMED-SECONDARY and the case has no house departure: G5a (chapter with DOI), G8 (news without author),
  G9 (n.d. with retrieval date) and G10 (web page).
- `structure`: the listed `elements_in_order` appear in order, the `must_not_contain` items are absent, and the
  links and reason codes match. The full `expected.reference` is the HOUSE rendering and does not gate. Every
  in-text citation is `structure`, because V-1b re-checked none of the in-text forms (V-1, Purdue only).

## Open questions recorded in cases (not gated)

- Whether the arXiv `[Preprint]` label is used (G4): APA's example has none and secondary blogs add it.
- Whether the software `[Computer software]` label is used (G12): secondary guides disagree.
- The ellipsis form for 21 or more authors (G3): `. . .` and `...` are both accepted.
- Single-editor `(Ed.)` (G6): not in the evidence record.
- A malformed identifier (H3): the plan says "omit" in its Validation rules and "plain text, no link" in its Safe
  links rules. This golden follows Validation and omits the identifier.

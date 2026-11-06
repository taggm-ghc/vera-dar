# Lanes enable checklist (NONASK-1103)

**Update 2026-11-05 (read from config at the working tree; DOCS-1105-REPO).** The text below is the original NONASK-1103 checklist and is kept. Current flag state differs for the news lane:
- News (current-reporting) lane: `enabled` in `config/reporting_lane.json` is **true** (owner decision 2026-11-05, quoted in that file: news sources yes, as far as each source's permissions and licence terms permit; GDELT only, `fallback_order` is `["gdelt"]`; `emit_link_only` true, `outlet_query` "all"). Only outlets with `terms_verdict` `allow` and a `terms_evidence` pointer are read in full (two at present: Global Voices, Wikinews; see `work/NEWS-LANE-1104/compliance.md`); every other outlet is title and link only (Deny All). The GDELT call and the real TLS-pinned path are **untested live** from the owner environment; offline tests only. `config/reporting_portals.json` still lists both portals as `draft` with `disclosure_ok` false; nothing in `vera/` reads those fields as of this pass (grep, 2026-11-05), so they do not gate the lane.
- Hacker News (Algolia) is not in the lane's fallback order; its terms status stays "unknown".
- Official-sources lane: built; H5 (OD-1106-HITL) added FEC, Census Bureau and EAC registry rows (status `owner_decided_unverified_paths`: robots.txt and paths not fetched) but the flag stays off until C1, C4, C5, C6, G8 and C7 are closed; `official_lane.enabled` and `official_lane.site_search` in `config/subquery.json` are **false**. The checklist below still applies before it is turned on. Registry verification work: `work/OFFICIAL-LANE-1104/` and `work/OFFICIAL-REGISTRY-VERIFY-1104/`.
- Because the news lane is on, `vera.disclosure.lanes_disclosure_on()` is true and the lane disclosure text is shown; items for the news lane in the list below were not individually re-checked or recorded as done.

Original text (superseded for the news lane, see above): both lane flags shipped OFF: `official_lane.enabled` in `config/subquery.json` and `enabled` in `config/reporting_lane.json`
(plus `disclosure_ok` / `state` in `config/reporting_portals.json`). Before enabling either, check and record every item:

- [ ] The security review has passed.
- [ ] A GDELT HTTP 200 response was observed from the owner environment (the agent IP gets 429s) and the artlist JSON field names were confirmed.
- [ ] The Hacker News search (Algolia) terms are confirmed.
- [ ] Vendor terms are re-checked: `terms_verdict` and `terms_evidence` in `config/entity_registry.json` (fetch only where `allow` and robots.txt permits; unknown means links only).
- [ ] The real TLS-pinned path (`default_transport` in `vera/public_fetch.py`) has been exercised once.
- [ ] The SD.1 disclosure renders (About, short notice, Ask notice) with the flag on (`vera.disclosure.lanes_disclosure_on()`).

Notes (orchestrator, R-042 honour-site-terms rule):

- HN Algolia terms status is "unknown" under R-042: no official API terms were found (`work/NONASK-1103/news-portals-evidence.md`) and the YC site terms ban robots without mentioning the API. This is a gate; the reporting lane stays OFF.
- Candidate alternative, NOT implemented: the Firebase HN API (MIT, documented "no rate limit", no keyword search).
- Outlet pages linked from GDELT/HN are links only unless the outlet is listed with `terms_verdict=allow` in `config/reporting_lane.json` `outlets` (empty by default).
- Security review `work/NONASK-1103/security-s6.md`: the BLOCKER and 4 SIGNIFICANT findings on `vera/public_fetch.py` must be closed (I-S6 fix merged) before enabling.

## Official lane site: query (separate switch)

- `config/subquery.json` `official_lane.site_search` ships `false`. When true, each allow-verdict entity gets one `site:<host> <terms>` query through the shared DuckDuckGo Lite provider (its cache, `daily_query_cap` and rate gate), after the research lane's DDG call. Rows marked links_only or unknown never get one.
- Before you turn it on, confirm the daily DDG cap has headroom for one extra query per entity, and check one live query from the owner environment.

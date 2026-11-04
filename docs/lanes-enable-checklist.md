# Lanes enable checklist (NONASK-1103)

Both lane flags ship OFF: `official_lane.enabled` in `config/subquery.json` and `enabled` in `config/reporting_lane.json`
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

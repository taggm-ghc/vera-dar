"""Read-time micro-topic key (ACQ S4-0 L2/L2b). No network and no writes (PD-3). Default: no DB (JSON seed).

Key order: (1) stored provenance.acquisition.micro_topic (K1/K4, top entry); (2) K4 best anchor by BM25 over a
frozen reference built from the anchor texts only (IDF snapshot from the anchors, never updated by arriving
documents) when score >= count_min_score.anchor_bm25.scholarly; (3) None = "unassigned". K5 (question-derived)
is disabled: no question-derived value is read, persisted or logged here.
Optional DB-backed registry (default off): vera.topic_registry.load_active_registry_data only, a SELECT in a READ ONLY
transaction (TOPICREG-1028); the DB result and the absent-table fallback are cached with a TTL (registry_table.cache_ttl_seconds).
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TOKEN = re.compile(r"[a-z0-9]+")
_K1, _B = 1.2, 0.75
_REG_CACHE: dict = {}
_DB_CACHE: dict = {}  # key -> (expires_at, Registry | None); None = table absent (JSON seed fallback), cached too
_now = time.monotonic
DEFAULT_DB_TTL = 300.0
DEFAULT_WINDOW = 2000
STORED_REGISTRY_SCHEMES = ("registry", "anchor_registry")


def tokens(text) -> list:
    return _TOKEN.findall(str(text or "").lower())


class Registry:
    """Frozen anchor registry + IDF snapshot (computed once from the anchor texts)."""

    def __init__(self, data: dict):
        self.version = str(data.get("anchor_registry_version") or "")
        self.entries = [e for e in (data.get("entries") or []) if isinstance(e, dict) and e.get("id")]
        self.by_id = {e["id"]: e for e in self.entries}
        self._tf = [Counter(tokens(e.get("anchor_text"))) for e in self.entries]
        self._len = [sum(c.values()) for c in self._tf]
        n = len(self.entries)
        self._avg = (sum(self._len) / n) if n else 0.0
        df: Counter = Counter()
        for c in self._tf:
            df.update(c.keys())
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, text: str) -> list:
        """[(score, entry)] for every anchor; document tokens are the query (unique terms); deterministic."""
        q = set(tokens(text))
        out = []
        for e, tf, dl in zip(self.entries, self._tf, self._len):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (_K1 + 1) / (f + _K1 * (1 - _B + _B * dl / (self._avg or 1.0)))
            out.append((s, e))
        return out

    def best(self, text: str):
        sc = self.scores(text)
        if not sc:
            return None
        return max(sc, key=lambda p: (round(p[0], 12), -self.entries.index(p[1])))


def _db_ttl(cfg) -> float:
    v = ((cfg or {}).get("registry_table") or {}).get("cache_ttl_seconds", DEFAULT_DB_TTL)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else DEFAULT_DB_TTL


def load_registry(cfg: dict | None = None, path=None, reader=None) -> Registry:
    """JSON seed by default. With registry_table.read_from_db true (default off) the DB table's active rows are used via
    vera.topic_registry (read-only); the seed only when the table is absent; any other problem raises (fail closed).
    The DB result, and the absent-table fallback, are cached for registry_table.cache_ttl_seconds (default 300; 0 = no cache),
    so a runtime activation becomes visible without a restart and an absent table is not re-queried on every call."""
    if path is None and ((cfg or {}).get("registry_table") or {}).get("read_from_db") is True:
        key = "db:" + str(_ROOT)
        hit = _DB_CACHE.get(key)
        now = _now()
        if hit is None or now >= hit[0]:
            from vera import topic_registry
            data = topic_registry.load_active_registry_data(cfg, reader)
            hit = (now + _db_ttl(cfg), None if data is None else Registry(data))
            _DB_CACHE[key] = hit
        return hit[1] if hit[1] is not None else _load_json_registry(cfg, path)
    return _load_json_registry(cfg, path)


def _load_json_registry(cfg, path) -> Registry:
    rel = path or (cfg or {}).get("registry_path") or "config/micro_topic_anchors.json"
    p = Path(rel)
    p = p if p.is_absolute() else _ROOT / p
    key = str(p)
    if key not in _REG_CACHE:
        with open(p, encoding="utf-8") as f:
            _REG_CACHE[key] = Registry(json.load(f))
    return _REG_CACHE[key]


def reset_for_tests() -> None:
    _REG_CACHE.clear()
    _DB_CACHE.clear()


def _threshold(cfg: dict) -> float:
    return float((((cfg.get("count_min_score") or {}).get("anchor_bm25") or {}).get("scholarly")))


def doc_text(title, text, cfg: dict) -> str:
    w = int((cfg or {}).get("text_window_chars") or DEFAULT_WINDOW)
    return f"{title or ''}\n{str(text or '')[:w]}"


def _prov_title(prov: dict):
    t = prov.get("title")
    if not isinstance(t, str):
        t = (prov.get("meta") or {}).get("title") if isinstance(prov.get("meta"), dict) else None
    return t if isinstance(t, str) else ""


def _entry_key(scheme, ident, score, reg: Registry | None, field=None) -> dict:
    """Granularity rule: a K4 entry with a parent OpenAlex topic counts toward openalex_topic:<parent>."""
    if scheme in STORED_REGISTRY_SCHEMES:
        e = (reg.by_id.get(ident) if reg else None) or {}
        field = field or e.get("field")
        parent = e.get("parent_openalex_topic")
        if parent and parent != "none":
            scheme, ident = "openalex_topic", parent
        else:
            scheme = "registry"
    out = {"scheme": scheme, "id": ident, "score": score}
    if field:
        out["field"] = field
    return out


def key_for_provenance(prov, text, cfg: dict):
    """{scheme, id, score, field?} or None (unassigned). Pure; never writes; text may be None (index time)."""
    if not isinstance(prov, dict):
        return None
    cfg = cfg or {}
    reg = None
    try:
        reg = load_registry(cfg)
    except (OSError, ValueError):
        reg = None
    stored = (prov.get("acquisition") or {}).get("micro_topic") if isinstance(prov.get("acquisition"), dict) else None
    if isinstance(stored, list) and stored and isinstance(stored[0], dict) and stored[0].get("id"):
        top = stored[0]
        if top.get("scheme") != "openalex_topic_provisional":  # K5 provisional keys are never read here
            return _entry_key(top.get("scheme"), str(top["id"]), top.get("score"), reg)
    if reg is None:
        return None
    body = doc_text(_prov_title(prov), text, cfg)
    if not body.strip():
        return None
    b = reg.best(body)
    if b is None or b[0] < _threshold(cfg):
        return None
    return _entry_key("registry", b[1]["id"], round(b[0], 4), reg, b[1].get("field"))


def key_id(k) -> str | None:
    return f"{k['scheme']}:{k['id']}" if k else None


def read_time_key(prov, text, cfg: dict):
    """L2b: key for an already-stored document (e.g. the 63 existing rows). Read-only; nothing is written."""
    return key_for_provenance(prov, text, cfg)


def _row_prov(row):
    p = row.get("provenance")
    if isinstance(p, (str, bytes)):
        try:
            p = json.loads(p)
        except ValueError:
            return None
    return p if isinstance(p, dict) else None


def row_key(row, cfg: dict) -> str | None:
    return key_id(key_for_provenance(_row_prov(row), row.get("content"), cfg))


def count_by_key(rows, cfg: dict) -> Counter:
    """Documents per key string; unassigned documents count toward nothing."""
    c: Counter = Counter()
    for r in rows or ():
        k = row_key(r, cfg)
        if k:
            c[k] += 1
    return c


def persisted_count(rows, key: str, cfg: dict) -> int:
    return count_by_key(rows, cfg).get(key, 0)


def k5_key(*_a, **_k):
    """K5 (question-derived OpenAlex topic query) is disabled in S4-0."""
    raise RuntimeError("micro_topic K5 is disabled in S4-0")


def k5_enabled(cfg: dict) -> bool:
    return False  # hard-off in S4-0 whatever the config says (L16 redaction lands in stage 1)

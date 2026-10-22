"""VERA-DAR: content-based domain identification (stdlib only; no LLM, DB, network or env access).

Method: TF-IDF cosine against per-domain profiles (centroids of length-normalised term frequencies; idf is applied at
scoring time with the CURRENT document-frequency table). Every threshold comes from the `identification` config block
(icfg); nothing is hardcoded here beyond the smoothing form of idf.

Derivation uses a minimum-support rule (amendment A1): a document that matches no domain is parked in a pending pool.
Similar pending documents form clusters; when a cluster reaches icfg["min_support"] documents ONE derived domain is
registered (capped by icfg["max_derived_domains"]) and every cluster document is associated with it.

classify() is read-only. Domain membership is a classification aid only, never an admission or quality signal.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from vera.domain_registry import normalize_domain_name

SECONDARY = ":secondary"  # asserted_by suffix of a SECONDARY document label (never changes a domain profile)


def tokenize(text: str, icfg: dict) -> list[str]:
    """casefold, icfg token_pattern, min_token_len, stopwords (order preserved, duplicates kept)."""
    pat = re.compile(icfg["token_pattern"])
    stop = set(icfg.get("stopwords") or [])
    min_len = int(icfg["min_token_len"])
    out = []
    for t in pat.findall(str(text or "").casefold()):
        if len(t) >= min_len and t not in stop:
            out.append(t)
    return out


def _tf(tokens: list[str]) -> dict[str, float]:
    n = len(tokens)
    if not n:
        return {}
    return {t: c / n for t, c in Counter(tokens).items()}


class DomainIdentifier:
    def __init__(self, registry, icfg: dict):
        self.registry = registry
        self.icfg = icfg
        self._n = 0
        self._df: Counter = Counter()
        self._profiles: dict[int, dict[str, float]] = {}
        self._doc_tf: dict[str, dict[str, float]] = {}
        self._results: dict[str, dict] = {}
        self._clusters: list[dict] = []  # pending: {"docs": [keys], "centroid": {term: w}}
        self._pending: list[str] = []  # pending documents (no domain yet); re-clustered from scratch on each park
        self._aliases: dict[int, int] = {}  # alias domain_id -> canonical domain_id (consolidation; rows are never deleted)
        self._pver = 0  # ACQ-REBUILD-PERF-1021: bumped at EVERY _profiles/_aliases mutation; keys the classify memo
        self._dmemo = None

    # ---- consolidation aliases ----
    @property
    def aliases(self) -> dict:
        """Read-only copy of the alias -> canonical mapping."""
        return dict(self._aliases)

    def _canon(self, did: int) -> int:
        return self._aliases.get(did, did)

    def _active(self) -> dict[int, dict[str, float]]:
        """Profiles of canonical (non-alias) domains only: classification, scoring, the ubiquity guard and the
        domain-level idf ignore alias profiles."""
        return {d: p for d, p in self._profiles.items() if d not in self._aliases}

    def merge_alias(self, alias_id: int, canonical_id: int) -> bool:
        """Fold ALIAS into CANONICAL: the alias profile (term sums) is added to the canonical profile, the alias is
        recorded and from then on ignored by classification. Idempotent; returns True when it changed anything.
        The registry is untouched (aliased domains stay as rows)."""
        alias_id, canonical_id = int(alias_id), self._canon(int(canonical_id))
        if alias_id == canonical_id or self._aliases.get(alias_id) == canonical_id:
            return False
        if alias_id in self._aliases:
            return False  # already aliased elsewhere
        self._sync_profiles()
        self._pver += 1
        prof = self._profiles.setdefault(canonical_id, {})
        for t, w in self._profiles.get(alias_id, {}).items():
            prof[t] = prof.get(t, 0.0) + w
        for a, c in list(self._aliases.items()):  # re-point aliases of the alias
            if c == alias_id:
                self._aliases[a] = canonical_id
        self._aliases[alias_id] = canonical_id
        return True

    # ---- scoring helpers ----
    def _idf(self, term: str) -> float:
        return math.log(1.0 + (self._n + 1.0) / (self._df.get(term, 0) + 1.0))

    def _vec(self, weights: dict[str, float]) -> tuple[dict[str, float], float]:
        v = {t: w * self._idf(t) for t, w in weights.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return v, norm

    def _cos(self, a: dict[str, float], b: dict[str, float]) -> float:
        va, na = self._vec(a)
        vb, nb = self._vec(b)
        if not na or not nb:
            return 0.0
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(x * vb.get(t, 0.0) for t, x in va.items()) / (na * nb)

    def _dcos(self, a: dict[str, float], b: dict[str, float], ctx: dict | None = None) -> float:
        """Class-based TF-IDF cosine (domain-level idf): once two or more domains exist, a term's weight is
        log(1 + D / number_of_domain_profiles_containing_it), so boilerplate present in every domain profile weighs
        little and topical terms concentrated in one domain weigh a lot. With fewer than two domains it is the plain
        document-level cosine."""
        # ACQ-REBUILD-PERF-1021: `ctx` is a memo for the READ-ONLY classify(), valid while _pver is unchanged (profiles
        # and aliases unchanged): the per-term domain factor and each profile's vector are computed once per
        # call instead of once per (text, profile) pair. Same expressions in the same order => bit-identical scores.
        profs = list(self._active().values()) if ctx is None else ctx["profs"]
        if len(profs) < 2:
            return self._cos(a, b)
        D = len(profs)
        fac = {} if ctx is None else ctx["fac"]
        def f(t):
            r = fac.get(t)
            if r is None:
                r = fac[t] = math.log(1.0 + D / max(1, sum(1 for pr in profs if t in pr)))
            return r
        def vec(w):
            v = {t: x * f(t) for t, x in w.items()}
            n = math.sqrt(sum(x * x for x in v.values()))
            return v, n
        va, na = vec(a)
        if ctx is None:
            vb, nb = vec(b)
        else:
            pv = ctx["pv"].get(id(b))
            if pv is None or pv[0] is not b:
                pv = ctx["pv"][id(b)] = (b, *vec(b))
            _, vb, nb = pv
        if not na or not nb:
            return 0.0
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(x * vb.get(t, 0.0) for t, x in va.items()) / (na * nb)

    def _salient(self, weights: dict[str, float]) -> dict[str, float]:
        """Shared-vocabulary view (D1): only terms whose document frequency lies in [salient_df_min, hi] with
        hi = min(max(salient_df_max_floor, frac * N), N - 1). A term present in EVERY document (df == N) carries no
        topical signal and is dropped (review HIGH-1). Current df, so it adapts as the corpus grows (no retune)."""
        ic = self.icfg
        hi = min(max(int(ic["salient_df_max_floor"]), int(float(ic["salient_df_max_frac"]) * self._n)), self._n - 1)
        lo = int(ic["salient_df_min"])
        return {t: w for t, w in weights.items() if lo <= self._df.get(t, 0) <= hi}

    def _salient_ready(self) -> bool:
        """The shared-vocabulary view is only meaningful once the corpus has assoc_salient_min_docs documents."""
        return self._n >= int(self.icfg["assoc_salient_min_docs"])

    def _salient_match(self, a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
        """(cosine, shared term count) over the salient views of a and b."""
        sa, sb = self._salient(a), self._salient(b)
        return self._cos(sa, sb), len(set(sa) & set(sb))

    def _sig(self, weights: dict[str, float]) -> set[str]:
        """Salient-term signature: the top sig_terms terms by weight x idf (current document frequencies), over terms
        seen in at least two documents once the corpus has sig_min_docs documents (a term unique to one document
        cannot be shared)."""
        ic = self.icfg
        v, _ = self._vec(weights)
        lo = 2 if self._n >= int(ic["sig_min_docs"]) else 1
        hi = max(int(ic["sig_df_max_floor"]), int(float(ic["sig_df_max_frac"]) * self._n))  # boilerplate: present in most documents
        ranked = sorted(((t, w) for t, w in v.items() if lo <= self._df.get(t, 0) <= hi), key=lambda x: (-x[1], x[0]))
        return {t for t, _w in ranked[: int(ic["sig_terms"])]}

    def _sig_pass(self, tf: dict[str, float], domain_id: int) -> bool:
        """True when the corpus is too small for signatures, or the document's salient-term signature shares at least
        sig_min_shared terms with the domain's (boilerplate shared by every domain is not in a signature)."""
        ic = self.icfg
        if self._n < int(ic["sig_min_docs"]):
            return True
        self._sync_profiles()
        return len(self._sig(tf) & self._sig(self._profiles.get(domain_id, {}))) >= int(ic["sig_min_shared"])

    def _doc_sim(self, tf: dict[str, float], profile: dict[str, float]) -> float:
        """Document-to-profile similarity; every ACCEPTED value is > 0 (callers test `> 0`), 0.0 when
        rejected. Two routes, the larger accepted value wins:
        * class-based whole-text cosine >= assoc_min_sim, with min_shared_salient shared terms that are not present in
          every domain profile (short, topically coherent documents);
        * once the corpus has sig_min_docs documents: SIGNATURE overlap. Accepted when the document's salient-term
          signature and the domain's share at least sig_min_shared terms and the cosine is at least sig_cos_floor; the
          value is shared/k (so a secondary-label ratio compares overlaps, not a constant offset). Long, lexically diverse documents of one subject have a low cosine
          (~0.2-0.5) but share 15-25% of their top terms; unrelated documents share almost none."""
        ic = self.icfg
        full = self._dcos(tf, profile)
        profs = list(self._active().values())
        guard = len(profs) >= int(ic["ubiquity_min_domains"])
        shared = sum(1 for t in tf if t in profile and (not guard or not all(t in pr for pr in profs)))
        best = full if full >= ic["assoc_min_sim"] and shared >= int(ic["min_shared_salient"]) else 0.0
        if self._n >= int(ic["sig_min_docs"]) and full >= float(ic["sig_cos_floor"]):
            sh = len(self._sig(tf) & self._sig(profile))
            if sh >= int(ic["sig_min_shared"]):
                best = max(best, sh / int(ic["sig_terms"]))
        return best

    def _sync_profiles(self) -> None:
        """Seed a profile for any registry domain lacking one, from its name tokens (additive; no config terms needed)."""
        for d in self.registry.list_domains():
            if d.domain_id not in self._profiles:
                # config/manual domains are seeded from their name tokens; a DERIVED domain starts empty (its profile is
                # exactly the sum of its member documents, the same definition live and after restore())
                self._profiles[d.domain_id] = {} if d.origin == "derived" else _tf(tokenize(d.name, self.icfg))
                self._pver += 1

    def _scores(self, tf: dict[str, float]) -> list[tuple[int, float]]:
        self._sync_profiles()
        sc = [(did, self._doc_sim(tf, prof)) for did, prof in self._active().items()]
        return sorted(sc, key=lambda x: (-x[1], x[0]))

    def _add_to_profile(self, did: int, tf: dict[str, float]) -> None:
        self._pver += 1
        prof = self._profiles.setdefault(self._canon(did), {})
        for t, w in tf.items():
            prof[t] = prof.get(t, 0.0) + w

    def _top_terms(self, weights: dict[str, float], n: int) -> list[str]:
        v, _ = self._vec(weights)
        return [t for t, _w in sorted(v.items(), key=lambda x: (-x[1], x[0]))[:n]]

    def _count_stats(self, tf: dict[str, float]) -> None:
        self._n += 1
        for t in tf:
            self._df[t] += 1

    # ---- public API ----
    def profile_terms(self, domain_id: int, n: int) -> list[str]:
        self._sync_profiles()
        return self._top_terms(self._profiles.get(domain_id, {}), n)

    def classify(self, text: str, kind: str = "chunk", exclude=()) -> list[dict]:
        """READ-ONLY: no registry write, no df update. kind in {"chunk","question"}. `exclude` removes domain ids
        BEFORE the floor/ratio filters (used to ask "which FOREIGN domain does this chunk resemble best")."""
        if kind == "question":
            floor, cap = self.icfg["question_min_sim"], self.icfg["max_question_domains"]
        elif kind == "chunk":
            floor, cap = self.icfg["chunk_min_sim"], self.icfg["max_domains_per_chunk"]
        else:
            raise ValueError("kind must be 'chunk' or 'question'")
        tf = _tf(tokenize(text, self.icfg))
        if not tf:
            return []
        self._sync_profiles()
        act = self._active()
        ctx = self._dmemo
        if ctx is None or ctx["ver"] != self._pver:  # memo valid until the next profile/alias mutation (see _dcos)
            ctx = self._dmemo = {"ver": self._pver, "profs": list(act.values()), "fac": {}, "pv": {}}
        sc = sorted(((did, self._dcos(tf, prof, ctx)) for did, prof in act.items() if did not in exclude), key=lambda x: (-x[1], x[0]))
        out = [{"domain_id": d, "score": s} for d, s in sc if s >= floor and s > 0]
        ratio = float(self.icfg["chunk_secondary_ratio" if kind == "chunk" else "question_secondary_ratio"])
        if out and ratio > 0:  # secondary labels must be comparable to the best one (boilerplate scores cannot qualify)
            out = [out[0]] + [x for x in out[1:] if x["score"] >= ratio * out[0]["score"]]
        return out[: int(cap)]

    def score_text(self, text: str, domain_id: int) -> float:
        """READ-ONLY whole-text cosine of `text` against ONE domain profile (D3 incremental re-classification)."""
        self._sync_profiles()
        tf = _tf(tokenize(text, self.icfg))
        return self._dcos(tf, self._profiles.get(domain_id, {})) if tf else 0.0

    def reclassify_docs(self, domain_id: int, asserted_by: str) -> list[str]:
        """D3: score every already-known document against ONE (new) domain only; associate those at or above
        assoc_min_sim (additive: nothing is ever removed; per-document cap max_domains_per_doc). Returns the
        doc_keys newly associated. Profiles are not changed."""
        ic = self.icfg
        self._sync_profiles()
        prof = self._profiles.get(domain_id, {})
        out = []
        for key in sorted(self._doc_tf):
            tf = self._doc_tf[key]
            have = self.registry.domains_for_corpus(key)
            if any(d.domain_id == domain_id for d in have) or len(have) >= int(ic["max_domains_per_doc"]):
                continue
            if self._doc_sim(tf, prof) > 0 and self.registry.associate_corpus(domain_id, key, asserted_by):
                out.append(key)
        return out

    def note_association(self, doc_key: str, domain_id: int) -> None:
        """A document gained a SECONDARY corpus association (chunk-derived or re-classification). A domain profile is
        the sum of the term frequencies of its PRIMARY member documents only (hits at ingest and promotions), so a
        weak secondary label can never pull a domain towards other content; restore() reproduces this from the
        ':secondary' asserted_by marker (D4)."""
        return None  # secondary associations never change a profile (no feedback loop); kept for API stability

    def restore_document(self, doc_key: str, text: str, domain_ids) -> None:
        """Rebuild stats for an already-known document: df/N and profiles only; no registry write, no derivation."""
        if doc_key in self._doc_tf:
            return
        toks = tokenize(text, self.icfg)
        if len(toks) < int(self.icfg["min_doc_terms"]):
            return  # live add_document neither counts nor profiles such a document
        tf = _tf(toks)
        self._doc_tf[doc_key] = tf
        self._count_stats(tf)
        self._sync_profiles()
        for did in sorted(set(domain_ids)):
            self._add_to_profile(did, tf)

    def add_document(self, doc_key: str, text: str, asserted_by: str) -> dict:
        if doc_key in self._results:
            return dict(self._results[doc_key], promoted=[])
        ic = self.icfg
        tokens = tokenize(text, ic)
        res = {"doc_key": doc_key, "domains": [], "created": [], "uncertain": False,
               "unclassified": True, "promoted": [], "pending": False}
        if len(tokens) < int(ic["min_doc_terms"]):
            self.registry.record_uncertain(doc_key, "too few terms to identify a domain", [])
            res["uncertain"] = True
            self._results[doc_key] = res
            return dict(res)
        tf = _tf(tokens)
        self._doc_tf[doc_key] = tf
        self._count_stats(tf)
        scores = self._scores(tf)
        hits = [(d, s) for d, s in scores if s > 0][: int(ic["max_domains_per_doc"])]
        if hits:
            # the best domain is PRIMARY (the document joins its profile); other domains are SECONDARY labels (bridge
            # documents) only when their overlap is comparable to the best one, and they never change a profile
            best = hits[0][1]
            for i, (d, s) in enumerate(hits):
                if i > 0 and (s < float(ic["secondary_ratio"]) * best or not self._sig_pass(tf, d)):
                    continue  # a secondary label needs comparable overlap AND, in a larger corpus, a signature match
                self.registry.associate_corpus(d, doc_key, asserted_by if i == 0 else asserted_by + SECONDARY)
                if i == 0:
                    self._add_to_profile(d, tf)
                res["domains"].append({"domain_id": d, "score": s})
            res["unclassified"] = False
            self._results[doc_key] = res
            if self._pending and self._salient_ready():  # the corpus grew: earlier pending documents may now cluster
                self._recluster(doc_key, asserted_by, res)
            return dict(res)
        self._park(doc_key, tf, scores, asserted_by, res)
        self._results[doc_key] = res
        return dict(res)

    # ---- pending pool and promotion ----
    def _park(self, doc_key, tf, scores, asserted_by, res) -> None:
        """A document that matches no domain is recorded once as an uncertain proposal. With min_support <= 1 (the
        shipped value) it founds a new DERIVED domain at once, which is PROVISIONAL (see DarCorpus.domain_status) until
        established_support documents belong to it; no ten-document gate and no manual registration. With
        min_support > 1 it stays pending and the pending pool is re-clustered from scratch once the corpus has
        assoc_salient_min_docs documents (the clustering route is exercised by tests; derivation is then a function of
        the pending SET, not of arrival order)."""
        ic = self.icfg
        cand = [self.registry.get(d).name for d, s in scores if s >= ic["uncertain_floor"] and self.registry.get(d)]
        self.registry.record_uncertain(doc_key, "no domain above association threshold; pending support", cand)
        res["uncertain"] = True
        res["pending"] = True
        self._pending.append(doc_key)
        if int(ic["min_support"]) <= 1:  # support 1: a document matching no domain founds one immediately
            self._clusters = [{"docs": [doc_key], "centroid": dict(tf)}]
            if self._promote(0, doc_key, asserted_by, res):
                self._pending = [k for k in self._pending if k != doc_key]
            self._clusters = []
        elif self._salient_ready():
            self._recluster(doc_key, asserted_by, res)

    def _recluster(self, current: str, asserted_by: str, res: dict) -> None:
        """Cluster ALL still-pending documents from scratch, in sorted key order, with the CURRENT document
        frequencies, so the outcome depends on the pending SET and not on arrival history (permutation stability).
        A cluster that reaches min_support is promoted to a derived domain and leaves the pool."""
        ic = self.icfg
        clusters: list[dict] = []
        for key in sorted(self._pending):
            tf = self._doc_tf[key]
            best_idx, best_sim = None, 0.0
            for i, c in enumerate(clusters):
                s, shared = self._salient_match(tf, c["centroid"])
                if s >= ic["pending_min_sim"] and shared >= int(ic["min_shared_salient"]) and s > best_sim:
                    best_idx, best_sim = i, s
            if best_idx is None:
                clusters.append({"docs": [key], "centroid": dict(tf)})
            else:
                c = clusters[best_idx]
                c["docs"].append(key)
                for t, w in tf.items():
                    c["centroid"][t] = c["centroid"].get(t, 0.0) + w
        for c in sorted((c for c in clusters if len(c["docs"]) >= int(ic["min_support"])),
                        key=lambda c: (-len(c["docs"]), c["docs"][0])):
            self._clusters = [c]
            if self._promote(0, current, asserted_by, res):
                gone = set(c["docs"])
                self._pending = [k for k in self._pending if k not in gone]
        self._clusters = []

    def _promote(self, idx: int, current: str, asserted_by: str, res: dict) -> bool:
        ic = self.icfg
        cluster = self._clusters[idx]
        cap = int(ic["max_derived_domains"])
        base = " ".join(self._top_terms(cluster["centroid"], int(ic["name_terms"])))
        if not base:
            return False
        derived = sum(1 for d in self.registry.list_domains() if d.origin == "derived")
        dom, created, n = None, False, 1
        while True:
            name = base if n == 1 else f"{base} ({n})"
            existing = self.registry.get(name)
            if existing is None:
                if derived >= cap:
                    return False  # cap reached: stay pending
                dom, created = self.registry.register(name, ic["derived_description"], origin="derived")
                break
            self._sync_profiles()
            if self._cos(cluster["centroid"], self._profiles.get(existing.domain_id, {})) >= ic["assoc_min_sim"]:
                dom = self.registry.get(self._canon(existing.domain_id)) or existing  # R5: same name AND similar content -> same domain
                break
            n += 1
        did = dom.domain_id
        self._sync_profiles()
        self._pver += 1
        if created:
            self._profiles[did] = {}
        for key in cluster["docs"]:
            self.registry.associate_corpus(did, key, asserted_by)
        for t, w in cluster["centroid"].items():
            self._profiles[did][t] = self._profiles[did].get(t, 0.0) + w
        for key in cluster["docs"]:
            r = res if key == current else self._results.get(key)
            if r is not None:
                r["domains"] = [{"domain_id": did, "score": self._cos(self._doc_tf[key], self._profiles[did])}]
                r["unclassified"] = False
                r["pending"] = False
            if key != current and key not in res["promoted"]:
                res["promoted"].append(key)
        if created:
            res["created"] = list(res.get("created") or []) + [did]
        self._clusters.pop(idx)
        return True

"""CORPUS-DIGEST-1104: scripted offline stand-in for the digest pipeline's model calls (no network, no database)."""
import re

from vera import corpus_description as cd


class FakeLLM:
    """Answers by step (identified by the system prompt). `clusters` = [(label, [1-based members])]. A sentence is
    a str (cites every cluster) or (text, [cluster ids]). `labels` maps a step to the model label it reports."""
    supports_avoid = True

    def __init__(self, clusters, sentences, coverage="", supported=None, fail_step=None, label="stub-model", labels=None,
                 title_ids="valid", derivations=None):
        """title_ids: "valid" (a title of the first cited cluster), "none", or "foreign" (a title of another cluster).
        derivations: {sentence text: [derivation, ...]} returned with that sentence's verdict."""
        self.title_ids, self.derivations = title_ids, derivations or {}
        self.clusters, self.sentences, self.coverage = clusters, sentences, coverage
        self.supported, self.fail_step, self.label, self.labels = supported, fail_step, label, labels or {}
        self.calls = []

    def step(self, messages):
        sys = messages[0]["content"]
        if sys.startswith(cd.CLUSTER_SYSTEM.split("{")[0]):
            return "cluster"
        if sys.startswith(cd.DRAFT_SYSTEM.split("{")[0]):
            return "draft"
        return "check"

    def __call__(self, messages, max_tokens=None, timeout_s=None, avoid=None):
        step = self.step(messages)
        self.calls.append({"step": step, "max_tokens": max_tokens, "timeout_s": timeout_s, "messages": messages,
                           "avoid": avoid})
        label = self.labels.get(step, self.label)
        if step == self.fail_step:
            raise RuntimeError("down")
        if step == "cluster":
            return {"clusters": [{"label": lab, "members": m} for lab, m in self.clusters]}, label
        if step == "draft":
            every = list(range(1, len(self.clusters) + 1))
            return {"sentences": [{"text": x, "clusters": every} if isinstance(x, str) else {"text": x[0], "clusters": x[1]}
                                  for x in self.sentences], "coverage": self.coverage}, label
        ctx, user = messages[1]["content"].split("\nSentences:\n", 1)
        tmap, cur = {}, None  # cluster id -> title ids shown in the context
        for line in ctx.splitlines():
            m = re.match(r"- C(\d+) ", line)
            if m:
                cur = int(m.group(1))
                tmap[cur] = []
            m = re.match(r"\s+\* T(\d+) ", line)
            if m and cur is not None:
                tmap[cur].append(int(m.group(1)))
        verdicts = []
        for line in user.splitlines():
            m = re.match(r"(\d+)\. \[([^\]]*)\] (.*)", line)
            if not m:
                continue
            n, cited, text = int(m.group(1)), [int(x) for x in re.findall(r"C(\d+)", m.group(2))], m.group(3)
            ok = self.supported is None or n in self.supported
            if self.title_ids == "none":
                ids = []
            elif self.title_ids == "foreign":
                other = [c for c in tmap if c not in cited]
                ids = tmap[other[0]][:1] if other else []
            else:
                ids = tmap.get(cited[0], [])[:1] if cited else []
            v = {"n": n, "supported": ok, "title_ids": ids}
            d = self.derivations.get(text) if isinstance(self.derivations, dict) else None
            if d is not None:
                v["derivations"] = d
            verdicts.append(v)
        return {"verdicts": verdicts}, label

    def steps(self):
        return [c["step"] for c in self.calls]

"""H19-FIX-1109 (design T5r / G7): a cited ND quote renders in quotation marks with the outlet, the title link, the
licence link and "Changes: excerpted"; a quote without a licence shows no licence link. Pure render, fake st."""
from vera import ui_evidence as ue


class St:
    def __init__(self):
        self.out = []

    def markdown(self, s, **k):
        self.out.append(("md", s))

    def caption(self, s, **k):
        self.out.append(("cap", s))

    def __getattr__(self, name):  # any other streamlit call is a no-op here
        return lambda *a, **k: None


def _data(att):
    return {"citations": [{"n": 2, "claim_id": 1, "verdict": "supported", "quote": "County officials certified the results.",
                           "quote_span": "chars=0-40", "span": "chars=0-389", "canonical_source_id": "https://www.votebeat.org/a",
                           "attribution": att}],
            "sources": [{"n": 2, "title": "Midterm results certified", "url": "https://www.votebeat.org/a"}]}


ND_ATT = {"creator": "As reported by votebeat.org", "title": "Midterm results certified",
          "source": "https://www.votebeat.org/a", "licence": "CC BY-NC-ND 4.0",
          "licence_url": "https://creativecommons.org/licenses/by-nc-nd/4.0/"}


def test_nd_quote_renders_quote_outlet_title_licence_and_changes():
    st = St()
    ue.render_citations(st, _data(ND_ATT))
    md = " ".join(s for k, s in st.out if k == "md")
    cap = " ".join(s for k, s in st.out if k == "cap").replace("\\", "")  # markdown escapes removed
    assert "\u201cCounty officials certified the results" in md and "\u201d (passage" in md
    assert "As reported by votebeat.org" in cap
    assert "https://www.votebeat.org/a" in cap
    assert "CC BY-NC-ND 4.0" in cap and "https://creativecommons.org/licenses/by-nc-nd/4.0/" in cap
    assert "Changes: excerpted" in cap


def test_no_licence_no_licence_link():
    st = St()
    ue.render_citations(st, _data({k: v for k, v in ND_ATT.items() if not k.startswith("licence")}))
    cap = " ".join(s for k, s in st.out if k == "cap").replace("\\", "")  # markdown escapes removed
    assert "creativecommons.org" not in cap and "CC BY" not in cap

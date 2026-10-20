"""Item 72 W4: licence gate aligned to R1's rule (reject ND / no-educational-use / fee-only; NC accepted)."""
import json

import pytest

from vera.licence_gate import RULES_PATH, apply_licence_gate, licence_decision, reject_reason


def c(content=None, url=None, meta=None, kind="full_content", page="https://x.org/a", gate="fetch"):
    lic = {"metadata": {"id": meta, "url": None} if meta else {},
           "content": {"id": content, "url": url} if (content or url) else {}}
    return {"url": page, "licence": lic, "stored_text_kind": kind, "gate_a_decision": gate, "gate_a_rationale": "r"}


@pytest.mark.parametrize("lid", ["CC-BY-NC-4.0", "CC BY-NC 4.0", "cc-by-nc-sa-4.0", "CC-BY-NC-SA", "by-nc", "CC-BY-4.0", "MIT"])
def test_nc_and_open_allowed(lid):
    assert licence_decision(c(content=lid))[0] == "allow"


@pytest.mark.parametrize("lid", ["CC-BY-ND-4.0", "CC BY-NC-ND 4.0", "CC-BY-NC-ND-4.0", "cc-by-nd", "CC_BY_NC_ND_4.0"])
def test_nd_rejected_id_variants(lid):
    d, why = licence_decision(c(content=lid))
    assert d == "reject" and "ND" in why


def test_url_variants():
    ok = "https://creativecommons.org/licenses/by-nc/4.0/"
    assert licence_decision(c(url=ok, content="CC-BY-NC-4.0"))[0] == "allow"
    assert licence_decision(c(url=ok))[0] == "defer"  # URL without an id is not a declared id (unchanged)
    assert licence_decision(c(url="HTTPS://CreativeCommons.org/licenses/BY-NC-ND/4.0/", content="x"))[0] == "reject"
    assert licence_decision(c(url="https://creativecommons.org/licenses/by-nd/3.0/"))[0] == "reject"


@pytest.mark.parametrize("lid", ["No Educational Use", "not-for-educational-use", "Educational use prohibited",
                                 "Fee Only", "subscription required", "PAYWALLED"])
def test_marker_rejects(lid):
    assert licence_decision(c(content=lid))[0] == "reject"


def test_missing_or_unknown_defers():
    assert licence_decision(c())[0] == "defer"
    assert licence_decision(c(meta="CC0-1.0"))[0] == "defer"            # metadata licence never covers content
    assert licence_decision(c(meta="CC0-1.0", kind="abstract_metadata"))[0] == "allow"
    assert licence_decision({"url": "https://x.org", "licence": None})[0] == "defer"


def test_nd_on_metadata_record_rejects_even_if_content_open():
    assert licence_decision(c(meta="CC-BY-ND-4.0", content="CC-BY-4.0"))[0] == "reject"


def test_arxiv_rule_unchanged():
    assert licence_decision(c(content="CC-BY-NC-4.0", page="https://arxiv.org/pdf/2401.1"))[0] == "reject"
    assert licence_decision(c(content="CC-BY-NC-4.0", page="https://arxiv.org/abs/2401.1"))[0] == "allow"


def test_gate_a_reject_never_overridden_and_defer_overrides_fetch():
    cs = apply_licence_gate([c(gate="reject", content="CC-BY-4.0"), c(content="CC-BY-NC-4.0"),
                             c(content="CC-BY-ND-4.0"), c()])
    assert [x["gate_a_decision"] for x in cs] == ["reject", "fetch", "reject", "defer"]
    cs = apply_licence_gate([c(gate="reject", content="CC-BY-NC-4.0")])
    assert cs[0]["gate_a_decision"] == "reject"


def test_rules_live_in_config_and_no_nc_term():
    r = json.loads(RULES_PATH.read_text())
    assert {"nd_tokens", "no_educational_use_markers", "fee_only_markers"} <= set(r)
    assert "nc" not in r["nd_tokens"]
    assert reject_reason("CC-BY-NC-4.0") is None


def test_missing_config_fails_verbosely(tmp_path):
    from vera.licence_gate import _load_rules
    with pytest.raises(RuntimeError, match="licence rules unreadable"):
        _load_rules(tmp_path / "nope.json")

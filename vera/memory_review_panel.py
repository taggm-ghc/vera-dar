"""Item #84 D6: operator review of pending memory claims (Streamlit, local/dev mode only).

The caller renders this only when VERA-DAR is NOT in public mode. The owner key is typed by the operator into a
password field (never prefilled, never read from the environment, never rendered); it is attached to the request
here and nowhere else. Claim text and source titles are untrusted and shown through the same safe renderers as
the answer. Errors show the status code only (no URL, no host, no response body).
"""
import requests

TIMEOUT_S = 15


def _headers(owner_key: str) -> dict:
    return {"X-Owner-Key": owner_key}


def fetch_pending(base_url: str, owner_key: str, timeout: int = TIMEOUT_S):
    """(status_code, list of pending claims or None)."""
    r = requests.get(f"{base_url.rstrip('/')}/memory/pending", headers=_headers(owner_key), timeout=timeout)
    return r.status_code, (r.json().get("pending") if r.status_code == 200 else None)


def decide(base_url: str, owner_key: str, claim_id: int, action: str, timeout: int = TIMEOUT_S) -> int:
    if action not in ("confirm", "reject"):
        raise ValueError("action must be confirm or reject")
    r = requests.post(f"{base_url.rstrip('/')}/memory/{int(claim_id)}/{action}", headers=_headers(owner_key),
                      timeout=timeout)
    return r.status_code


def render_memory_review(st, base_url: str) -> None:
    from vera.sources_sidebar import _link, plain
    from vera.ui_safety import safe_markdown

    with st.sidebar.expander("Operator: memory review (dev mode)"):
        owner_key = st.text_input("Owner key (X-Owner-Key)", value="", type="password", key="vera_owner_key_input",
                                  help="Sent only to the API base URL above. Never prefilled.")
        if not owner_key:
            st.caption("Type the owner key to list claims waiting for review.")
            return
        try:
            code, pending = fetch_pending(base_url, owner_key)
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {type(exc).__name__}")
            return
        if code != 200:
            st.error(f"Could not list pending claims (HTTP {code}).")
            return
        if not pending:
            st.caption("Nothing waiting for review.")
            return
        for item in pending:
            cid = int(item["claim_id"])
            st.markdown(safe_markdown(item.get("claim_text", "")))
            st.caption(f"#{cid} · {plain(item.get('state') or '')} · {plain(item.get('created_at') or '')}")
            for s in item.get("sources") or []:
                st.caption(_link(s.get("title") or s.get("url") or "", s.get("url") or ""))
            c1, c2 = st.columns(2)
            for col, action in ((c1, "confirm"), (c2, "reject")):
                if col.button(action.capitalize(), key=f"mem_{action}_{cid}"):
                    try:
                        status = decide(base_url, owner_key, cid, action)
                    except requests.exceptions.RequestException as exc:
                        st.error(f"Request failed: {type(exc).__name__}")
                    else:
                        (st.success if status == 200 else st.error)(f"{action}: HTTP {status}")

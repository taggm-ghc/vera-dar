"""Item #83 (ported from AI-Internship's api_client.triage_api): when the UI's call to the API fails, walk the
request path in order and stop at the earliest failing step. Messages never contain the host (the live
deployment URL must not appear on a public page or in screenshots)."""
import requests

TRIAGE_TIMEOUT_S = 65.0  # a Render free-tier wake can take up to a minute


def _is_edge_bounce(resp) -> bool:
    """Render's own wake-up bounce: a bare 429 without VERA-DAR's JSON body (VERA-DAR's caps answer JSON with detail)."""
    if resp.status_code != 429:
        return False
    try:
        return not isinstance(resp.json(), dict)
    except ValueError:
        return True


def triage_api(base_url: str, failing_path: str, http_get=None) -> list[dict]:
    """Steps: address configured -> reachable -> awake -> /health ok -> failing path retried. Each step is
    {step, ok, detail}; the list stops at the first failure."""
    get = http_get or (lambda url: requests.get(url, timeout=TRIAGE_TIMEOUT_S))
    steps: list[dict] = []

    def add(step, ok, detail):
        steps.append({"step": step, "ok": ok, "detail": detail})
        return ok

    base = (base_url or "").strip().rstrip("/")
    usable = base.lower().startswith(("http://", "https://"))
    if not add("1. API address configured", usable, "set" if usable else
               "no usable API address: the UI service needs VERA_API_BASE_URL"):
        return steps
    try:
        r = get(f"{base}/health")
    except requests.ConnectionError:
        add("2. API reachable", False, "nothing answered at the configured address (wrong address, or the API "
            "service is stopped or suspended)")
        return steps
    except requests.Timeout:
        add("2. API reachable", True, "the address accepted the connection")
        add("3. API awake", False, f"no reply within {TRIAGE_TIMEOUT_S:.0f}s; the API may still be waking up - "
            "wait a minute and ask again")
        return steps
    except requests.RequestException as exc:
        add("2. API reachable", False, f"request failed ({type(exc).__name__})")
        return steps
    add("2. API reachable", True, "the address answered")
    if _is_edge_bounce(r) or r.status_code in (502, 503, 504):
        add("3. API awake", False, f"the hosting platform answered HTTP {r.status_code} for the API: it is "
            "still starting - ask again in a minute")
        return steps
    add("3. API awake", True, "replied")
    try:
        healthy = r.status_code == 200 and r.json().get("status") == "ok"
    except (ValueError, AttributeError):
        healthy = False
    if not add("4. API healthy", healthy, "/health ok" if healthy else
               f"/health returned HTTP {r.status_code}; the API is running but unhealthy"):
        return steps
    if failing_path:
        try:
            again = get(f"{base}{failing_path}")
            code = again.status_code
            add(f"5. {failing_path} reachable", code not in (404, 502, 503, 504),
                "the endpoint answers (your request may have failed while the API was waking - ask again)"
                if code not in (404, 502, 503, 504) else
                "HTTP 404: the deployed API has no such endpoint - it is probably older than this UI" if code == 404
                else f"HTTP {code} from the endpoint while /health is ok")
        except requests.RequestException as exc:
            add(f"5. {failing_path} reachable", False, f"failed again ({type(exc).__name__})")
    return steps


def earliest_failure(steps: list[dict]) -> dict | None:
    return next((s for s in steps if not s["ok"]), None)

"""PORTAL-WIRE-A-1110: test helper. Install the real host_roles.json plus extra test-only ORIGINAL / PORTAL hosts."""
import json

from vera import host_role


def install(original=(), portal=(), repo=(), drop=(), drop_doi_rules=False):
    cfg = json.loads(host_role.DEFAULT_PATH.read_text(encoding="utf-8"))
    cfg["hosts"] = [r for r in cfg["hosts"] if r["match"] not in drop]
    cfg["host_patterns"] = [r for r in cfg.get("host_patterns", []) if not any(d in r.get("regex", "") for d in drop)]
    if drop_doi_rules:
        cfg["doi_prefix_roles"] = []
    for role, hosts in (("ORIGINAL", original), ("PORTAL", portal), ("REPOSITORY_OF_RECORD", repo)):
        for h in hosts:
            cfg["hosts"].append({"match": h, "role": role, "kind": "publisher", "basis": "test-only host"})
    host_role.set_registry_for_tests(host_role.Registry(cfg, "test"))


def reset():
    host_role.set_registry_for_tests(None)

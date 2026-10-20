"""VERA-DAR: consolidation of near-duplicate DERIVED domains (stdlib only; pure planning, no registry writes).

Order effects at ingest can leave several derived domains for one subject. plan_consolidation() proposes which of them
are the same: two derived domains merge when their profiles share at least icfg["consolidate_min_shared"] signature
terms AND their plain cosine is at least icfg["consolidate_min_cos"]; groups are the transitive closure and the
canonical domain of a group is its lowest domain_id. The caller applies the plan with
DomainIdentifier.merge_alias(alias, canonical); rows are never deleted. All thresholds come from config."""
from __future__ import annotations


def plan_consolidation(identifier, registry, icfg: dict) -> list[dict]:
    """[{"canonical": int, "alias": int, "shared": int, "cos": float}] sorted by alias id; [] when disabled."""
    if icfg.get("consolidate_enabled") is not True:
        return []
    min_shared, min_cos = int(icfg["consolidate_min_shared"]), float(icfg["consolidate_min_cos"])
    identifier._sync_profiles()
    aliased = identifier.aliases
    ids = sorted(d.domain_id for d in registry.list_domains()
                 if d.origin == "derived" and d.domain_id not in aliased and identifier._profiles.get(d.domain_id))
    sigs = {i: identifier._sig(identifier._profiles[i]) for i in ids}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    stats: dict[tuple[int, int], tuple[int, float]] = {}
    for ai, a in enumerate(ids):
        for b in ids[ai + 1:]:
            shared = len(sigs[a] & sigs[b])
            if shared < min_shared:
                continue
            cos = identifier._cos(identifier._profiles[a], identifier._profiles[b])
            if cos < min_cos:
                continue
            stats[(a, b)] = (shared, cos)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)  # the root is always the lowest id of the group
    plan = []
    for i in ids:
        root = find(i)
        if root == i:
            continue
        direct = stats.get((root, i)) or max((v for (x, y), v in stats.items() if i in (x, y)), key=lambda v: (v[1], v[0]))
        plan.append({"canonical": root, "alias": i, "shared": int(direct[0]), "cos": float(direct[1])})
    return plan

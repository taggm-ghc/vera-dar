"""VERA-DAR safety core (PLAN-ENV v2.1, ENV-B): endpoint classification, connect guard, SQL guard, destructive guard.

Guard files from creation (plan §6). Interface: frozen in the owner's private implementation notes (not published). The package root imports nothing,
so `safety.endpoint_class` stays stdlib-only when loaded by path or by package.
"""

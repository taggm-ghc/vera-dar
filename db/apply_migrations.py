"""VERA-DAR migration tool, fail-closed (P1-MIG staged patch, NOT INTEGRATED; hazards HZ-006-RB, HZ-DAR-1, N-1 stay OPEN).

Operations are separate subcommands; nothing connects unless the operation needs it:

    python db/apply_migrations.py plan 005 006                    # never connects; env files are parsed only to
                                                                  # classify the target, password values dropped
    python db/apply_migrations.py plan --approve-sha 005=<sha256> 005   # also checks a binding before review sign-off
    python db/apply_migrations.py --dry-run 005                   # alias of `plan` (old flag; now never connects)
    python db/apply_migrations.py inspect --target local          # read-only status (connects, read-only txn)
    python db/apply_migrations.py rehearse --target local 008     # execute then roll back; LOCAL + marker only
    python db/apply_migrations.py apply --target local 008
    python db/apply_migrations.py apply --target production --owner-approval OD-XXXX --approve-sha 008=<sha256> 008
        (also needs VERA_MIGRATE_PRODUCTION=I_ACCEPT_OD-XXXX in the process env only, a TTY, a typed
         confirmation, and the admin password typed at a getpass prompt; refused if DB_ADMIN_PASSWORD
         is in the process env or in any env file the tool can see, if any DB URL carries a password, or if any
         PG* variable is set; --approve-sha must equal the sha256 of the reviewed file, so swapped bytes are refused)
    python db/apply_migrations.py rollback 006                    # INSPECT ONLY: never executes a rollback

K18W INC-3: each per-file APPLY transaction starts with SET TRANSACTION READ WRITE (the only write opt-in, issued
after every tool-side gate; plan, --dry-run, inspect, rehearse and rollback never opt in). Only the pinned 008
bootstrap (name + sha256) is accepted while tracking is absent. A failed COMMIT is reported as OUTCOME UNKNOWN.

PLAN-ENV (owner decisions 2026-10-19): env files live under R/config/env (.env-intern, .env-vera for fingerprints only;
.env-local-pg and .test-db-marker as path names). D-ENV-5 = DEFER: the production path below ships DISABLED
(PRODUCTION_PATH_ENABLED = False, refusal "D-ENV-5-pending"). A connecting local op needs a verified-disposable database,
which is unreachable, so it refuses with DISPOSABLE-VERIFY-DISABLED-pending-isolation-review; `plan` and `rollback` stay
offline. With no production fingerprint sources every connecting op is refused. Migrations are never run from here today.

Ids are exactly three digits; only top-level migrations/NNN_vera_*.sql files are reachable.
This module imports no DB driver at import time and never imports week-1v2 db.py or dotenv.
Credentials and host names are never printed.
"""
from __future__ import annotations

import getpass as _getpass  # K18W (QA F17): bound before HERE is put on sys.path, so db/getpass.py cannot shadow it
import os
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # own dir only (no cross-repo path, V-4)
import migration_guard as g  # noqa: E402  (stdlib only)
# PLAN-ENV E3: env file paths come from vera.env_files (R/config/env). R/vera is appended (never prepended) so it cannot
# shadow db/ modules; vera.env_files only defines paths and imports vera.env_precedence (no DB driver, no I/O).
_VERA_ROOT = HERE.parent
if str(_VERA_ROOT) not in sys.path:
    sys.path.append(str(_VERA_ROOT))
from vera import env_files as _ef  # noqa: E402

OPS = ("plan", "inspect", "rehearse", "apply", "rollback")
VALUE_FLAGS = ("--target", "--owner-approval", "--allow-destructive")   # K18W (QA F7): unused --env-file removed
MULTI_FLAGS = ("--approve-sha",)   # r3 (SEC S-3): repeated NNN=<64 hex>, binds the approval to reviewed bytes
# K18W (QA F7): options each op accepts; anything else is refused, never silently ignored.
OP_FLAGS = {
    "plan": {"--approve-sha"},
    "rollback": set(),
    "inspect": {"--target"},
    "rehearse": {"--target", "--approve-sha", "--allow-destructive", "--owner-approval", "--fresh-install"},
    "apply": {"--target", "--approve-sha", "--allow-destructive", "--owner-approval", "--fresh-install"},
}
OPT_ECHO_RE = g.re.compile(r"--[a-z][a-z0-9_-]{0,22}|-[A-Za-z]")   # echo only option-shaped words
EXIT_INTERRUPTED = 130
LOCK_KEY = 0x56455241  # 'VERA'
LOCK_TIMEOUT, STATEMENT_TIMEOUT = "5s", "60s"   # UNCALIBRATED (V-8): calibration note required before production use
# r3 (SEC S-5, V QA-1): allowlist-based libpq env rule on EVERY connecting path: any PG* variable not listed here
# is refused. The allowlist is deliberately empty.
PG_ENV_ALLOW: frozenset = frozenset()
# K18W INC-3 (WS-8a F-5): ambient Kerberos/GSS client configuration is refused by name on every connecting path
# (PGSSLCERT/PGSSLKEY/PGGSSLIB/PGKRBSRVNAME/PGGSSENCMODE are already refused by the PG* rule above).
AMBIENT_AUTH_PREFIXES = ("PG", "KRB5", "GSS")
WRITE_OPT_IN = "SET TRANSACTION READ WRITE"   # K18W INC-3 (WS-8b C1): first statement of each per-file apply txn
SHA_ARG_RE = g.re.compile(r"([0-9]{3})=([0-9a-f]{64})")
# PLAN-ENV E3p / D-ENV-5 (owner decision 2026-10-19 = DEFER): the owner-only human production path stays in the code
# but ships DISABLED. No env var, flag or argument enables it; enabling is a one-line, owner-approved code change.
PRODUCTION_PATH_ENABLED = False
D_ENV_5_PENDING = "D-ENV-5-pending"
AGENT_MARKERS = ("CLAUDECODE", "AI_AGENT")
# PLAN-ENV C2 + owner decision 2026-10-19: no disposable verification is reachable, so any connecting local op refuses.
VERIFY_DISABLED = "DISPOSABLE-VERIFY-DISABLED-pending-isolation-review"
NO_SOURCES = "no_sources"
# Names only: neither file is opened for classification of "disposable" (the marker is never read at all).
LOCAL_ENV_NAME, MARKER_NAME = ".env-local-pg", ".test-db-marker"
LOCAL_ENV_FILE, MARKER_FILE = _ef.ENV_DIR / LOCAL_ENV_NAME, _ef.ENV_DIR / MARKER_NAME
# C3: inherited names removed from the process env after fingerprinting (same set as env_precedence's STRIP set).
try:
    from vera import env_precedence as _ep  # noqa: E402
    STRIP_PATTERNS: tuple = tuple(_ep._INHERITED_STRIP)
except Exception:  # noqa: BLE001  fail closed to a floor set
    STRIP_PATTERNS = ("*DB_URL*", "DATABASE_URL", "PG*", "KRB5*", "GSS*", "*_DBADMIN*", "DB_ADMIN_*",
                      "VERA_MIGRATE_PRODUCTION")


def default_prod_env_files() -> list[Path]:
    """Fingerprint sources only (password values dropped by the parser): R/config/env/.env-intern and .env-vera."""
    return [_ef.INTERN_ENV, _ef.VERA_ENV]


def _real_install_guard(context: str) -> None:
    """Lazy import of the process-global connect guard (safety/connect_guard.py). Raises on any failure (fail closed)."""
    from safety import connect_guard  # noqa: PLC0415
    connect_guard.install(context)


class Ctx:
    """Injection seams so tests never touch a real DB, TTY or env file."""

    def __init__(self, *, migrations_dir: Path | None = None, prod_env_files: list[Path] | None = None,
                 local_env_file: Path | None = None, marker_file: Path | None = None,
                 environ: dict | None = None, connect: Callable | None = None, isatty: Callable | None = None,
                 prompt: Callable | None = None, getpass: Callable | None = None, out=None, err=None,
                 install_guard: Callable | None = None):
        self.migrations_dir = migrations_dir or HERE / "migrations"
        self.prod_env_files = prod_env_files if prod_env_files is not None else default_prod_env_files()
        self.local_env_file = local_env_file or LOCAL_ENV_FILE
        self.marker_file = marker_file or MARKER_FILE   # path name only; never read (owner decision 2026-10-19)
        self.install_guard = install_guard or _real_install_guard
        self.environ = os.environ if environ is None else environ
        self.connect = connect or _real_connect
        self.isatty = isatty or _tty_ok
        self.prompt = prompt or _tty_prompt
        self.getpass = getpass or _tty_getpass
        self.out = out or sys.stdout
        self.err = err or sys.stderr

    def say(self, msg: str):
        print(msg, file=self.out)


# ---------------------------------------------------------------- real I/O (never reached by tests)
def _tty_ok() -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        with open("/dev/tty"):
            return True
    except OSError:
        return False


def _tty_prompt(text: str) -> str:
    with open("/dev/tty", "r+") as t:
        t.write(text)
        t.flush()
        return t.readline().rstrip("\n")


def _tty_getpass(text: str) -> str:
    with open("/dev/tty", "w") as t:   # K18W (QA F17): closed after use
        return _getpass.getpass(text, stream=t)  # getpass reads /dev/tty itself


def connect_kwargs(ep: g.Endpoint, password: str, cls: str) -> dict:
    """libpq parameters. passfile=/dev/null: never fall back to ~/.pgpass/PGPASSFILE (r2). sslmode=require for every
    non-local target (r3, SEC S-5; verify-full needs the provider CA and is an owner decision).
    K18W INC-3 (WS-8a F-5): ambient client credentials are neutralised: no client certificate is sent
    (sslcertmode=disable), no GSS encryption (gssencmode=disable), and require_auth limits authentication to the
    typed password (production: scram-sha-256) or to no client credential at all (local: none), so a Kerberos ticket,
    ~/.postgresql client cert or trust/cert-only rule can never stand in for the typed password. Needs libpq >= 16;
    an older libpq rejects the keywords and the connect fails closed (UNVERIFIED against the real driver)."""
    kw = dict(host=ep.host, port=ep.port, dbname=ep.dbname, user=ep.role, password=password, passfile="/dev/null",
              connect_timeout=10, application_name="vera-apply-migrations",
              sslcertmode="disable", gssencmode="disable", require_auth="none" if cls == "local" else "scram-sha-256")
    if cls != "local":
        kw["sslmode"] = "require"
    return kw


def _real_connect(ep: g.Endpoint, password: str, cls: str = "production"):
    """Lazy driver import, reached only after every gate passed. Builds its own DSN (no week-1v2 db.py)."""
    import psycopg2  # noqa: PLC0415
    conn = psycopg2.connect(**connect_kwargs(ep, password, cls))
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------- argument parsing (strict; unknown flags refused)
def parse(argv: list[str]) -> tuple[str, dict, list[str]]:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        raise g.Refusal(__doc__.strip())
    if args[0] == "--dry-run":
        op = "plan"
        args = args[1:]
    elif args[0] in OPS:
        op = args.pop(0)
    else:
        raise g.Refusal(f"first argument must be one of {', '.join(OPS)} (or --dry-run)")
    opts: dict = {}
    ids: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in VALUE_FLAGS:
            if i + 1 >= len(args) or a in opts:
                raise g.Refusal(f"{a} needs exactly one value and may appear once")
            opts[a] = args[i + 1]
            i += 2
            continue
        if a in MULTI_FLAGS:
            if i + 1 >= len(args):
                raise g.Refusal(f"{a} needs a value")
            opts.setdefault(a, []).append(args[i + 1])
            i += 2
            continue
        if a == "--fresh-install":
            opts[a] = True
        elif a.startswith("-"):
            # old tool silently ignored typos such as --dryrun. K18W (QA F16): echo only a plain option word.
            shown = repr(a) if OPT_ECHO_RE.fullmatch(a) else f"(argument {i + 1}, not echoed)"
            raise g.Refusal(f"unknown option {shown}")
        else:
            ids.append(a)
        i += 1
    extra = sorted(set(opts) - OP_FLAGS[op])
    if extra:
        raise g.Refusal(f"{op} does not accept {', '.join(extra)}")   # K18W (QA F7): never accepted-and-ignored
    if "--target" in opts and opts["--target"] not in ("local", "production"):
        raise g.Refusal("--target must be local or production")
    if op in ("apply", "rehearse", "inspect") and "--target" not in opts:
        raise g.Refusal(f"{op} requires --target local|production")
    if op in ("plan", "apply", "rehearse", "rollback"):
        g.validate_prefixes(ids)
    elif ids:
        raise g.Refusal("inspect takes no migration ids")
    if op == "rollback" and len(ids) != 1:
        raise g.Refusal("rollback inspects exactly one id")
    return op, opts, ids


# ---------------------------------------------------------------- offline steps
def load_batch(ctx: Ctx, ids: list[str]) -> list[tuple[str, Path, bytes, g.Lint]]:
    batch = []
    for p in ids:
        f = g.resolve(ctx.migrations_dir, p)
        data = f.read_bytes()
        batch.append((p, f, data, g.lint_sql(f.name, data.decode("utf-8", errors="replace"))))
    return batch


def classify_targets(ctx: Ctx) -> tuple[str, g.Endpoint | None, set[str], set[str]]:
    """Classify the endpoint this run would use, offline. Returns (class, endpoint, prod_fps, secret_keys_seen)."""
    prod_fps, seen = g.production_fingerprints(ctx.prod_env_files)
    lkv, lseen = g.parse_env_file(ctx.local_env_file)
    seen |= lseen
    if lkv.get("LOCAL_PG_DB"):
        ep = g.endpoint(lkv.get("LOCAL_PG_HOST", "/var/run/postgresql"), lkv.get("LOCAL_PG_PORT", "5432"),
                        lkv["LOCAL_PG_DB"], lkv.get("LOCAL_PG_ROLE", ""))
        if ep.fingerprint() in prod_fps:
            raise g.Refusal("local env file names a production endpoint")
        return g.classify(ep, prod_fps), ep, prod_fps, seen
    return "unknown", None, prod_fps, seen


def prod_endpoint(ctx: Ctx) -> g.Endpoint | None:
    for p in ctx.prod_env_files:
        kv, _ = g.parse_env_file(p)
        if "DB_HOST" in kv and "DB_NAME" in kv:
            return g.endpoint(kv["DB_HOST"] + kv.get("DB_HOST_EXTERNAL_SUFFIX", ""), kv.get("DB_PORT", "5432"),
                              kv["DB_NAME"], kv.get("DB_ADMIN_ROLE", ""))
    return None


def print_plan(ctx: Ctx, op: str, batch, target_class: str):
    for p, f, data, lint in batch:
        ctx.say(f"FILE {f.name} sha256={g.sha256_hex(data)[:12]} bytes={len(data)}")
        for h in lint.hard:
            ctx.say(f"  REFUSE {h}")
        for d in lint.destructive:
            ctx.say(f"  DESTRUCTIVE {d} (needs --allow-destructive <same approval id>)")
        for a in lint.advisory:
            ctx.say(f"  ADVISORY {a}")
    ctx.say("LINT advisory pre-check only, not a safety certification; the gate is human review of the exact bytes "
            "(--approve-sha NNN=<sha256>)")
    ctx.say(f"TARGET computed-class={target_class} (offline; no connection made)")


def audit(ctx: Ctx, **kv):
    ctx.say("AUDIT " + " ".join(f"{k}={v}" for k, v in kv.items()))


# ---------------------------------------------------------------- gates
def production_gate(ctx: Ctx, op: str, opts: dict, batch, secrets_seen: set[str], ep: g.Endpoint | None) -> str:
    """Accident and audit controls (V-2 relabel); custody (g) is the only one that resists a determined caller."""
    if op != "apply":
        raise g.Refusal(f"{op} is never allowed on production")                     # (e)
    if opts.get("--fresh-install"):
        raise g.Refusal("--fresh-install is never allowed on production")           # (e)
    oid = opts.get("--owner-approval", "")
    if g.APPROVAL_RE.fullmatch(oid) is None:
        raise g.Refusal("production apply needs --owner-approval OD-<id>")          # (b)
    if "VERA_MIGRATE_PRODUCTION" in secrets_seen:
        raise g.Refusal("VERA_MIGRATE_PRODUCTION found in an env file (source class: env-file); process env only")  # V-6
    if ctx.environ.get("VERA_MIGRATE_PRODUCTION") != f"I_ACCEPT_{oid}":
        raise g.Refusal("VERA_MIGRATE_PRODUCTION does not repeat the approval id")   # (c)
    if "DB_ADMIN_PASSWORD" in ctx.environ or "DB_ADMIN_PASSWORD" in secrets_seen:
        raise g.Refusal("DB_ADMIN_PASSWORD present in process env or an env file; production needs owner custody (V-2(g))")
    file_hits = sorted(k for k in secrets_seen if k != "VERA_MIGRATE_PRODUCTION")
    env_hits = g.env_secret_hits(ctx.environ) + g.env_url_passwords(ctx.environ)
    if file_hits or env_hits:
        where = "; ".join(x for x in (f"env-file: {','.join(file_hits)}" if file_hits else "",
                                       f"process-env: {','.join(sorted(set(env_hits)))}" if env_hits else "") if x)
        raise g.Refusal(f"password material embedded in a DB URL, DSN or password-named key ({where}); values "
                        "suppressed; production needs owner custody (SEC S-2)")
    check_bindings(opts, batch, required=True)
    for p, f, data, lint in batch:
        if lint.hard:
            raise g.Refusal(f"{f.name}: hard refusal ({lint.hard[0]})")
        if lint.destructive and opts.get("--allow-destructive") != oid:
            raise g.Refusal(f"{f.name}: destructive statements need --allow-destructive {oid}")   # (f)
    if not ctx.isatty():
        raise g.Refusal("production apply needs an interactive terminal")          # (d)
    # K18W (QA F11): confirm the SAME endpoint object that will be connected (no second env-file read)
    if ep is None or not ep.role:
        raise g.Refusal("production endpoint not configured")
    typed_db = ctx.prompt("Type the database name to confirm: ")
    typed_id = ctx.prompt("Type the owner approval id: ")
    if typed_db != ep.dbname or typed_id != oid:
        raise g.Refusal("typed confirmation does not match")                        # (d)
    return oid


def sha_bindings(opts: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in opts.get("--approve-sha", []):
        m = SHA_ARG_RE.fullmatch(v)
        if m is None or m.group(1) in out:
            raise g.Refusal("--approve-sha must be NNN=<64 lowercase hex>, once per id")
        out[m.group(1)] = m.group(2)
    return out


def check_bindings(opts: dict, batch, required: bool):
    """SEC S-3 sha binding. K18W (QA F7): when --approve-sha is given it is enforced on EVERY op (plan, local
    apply/rehearse), never accepted-and-ignored; on production it is required for every file."""
    bound = sha_bindings(opts)
    if not bound and not required:
        return
    unused = sorted(set(bound) - {b[0] for b in batch})
    if unused:
        raise g.Refusal(f"--approve-sha given for migration(s) not in this run: {','.join(unused)}")
    for p, f, data, lint in batch:
        if bound.get(p) != g.sha256_hex(data):
            raise g.Refusal(f"{f.name}: --approve-sha {p}=<sha256 of the reviewed file> missing or does not match "
                            "(the approval is bound to the exact reviewed bytes, SEC S-3)")


def check_env_credentials(ctx: Ctx, local_file_secrets: set[str]):
    """K18W (QA F3/F10): on EVERY connecting path, before any prompt or password request: refuse password
    material in the process env (and, for local, in the local env file). Names only, values never kept."""
    hits = sorted(set(g.env_secret_hits(ctx.environ) + g.env_url_passwords(ctx.environ)))
    if hits:
        raise g.Refusal(f"password material in the process env (process-env: {','.join(hits)}); values suppressed; unset it "
                        "before any connecting run")
    lf = sorted(k for k in local_file_secrets if k != "VERA_MIGRATE_PRODUCTION")
    if lf:
        raise g.Refusal(f"password material in the local env file ({','.join(lf)}); local runs never use a password")


def check_libpq_env(ctx: Ctx):
    bad = sorted(k for k in ctx.environ if k.upper().startswith(AMBIENT_AUTH_PREFIXES) and k not in PG_ENV_ALLOW)
    if bad:
        raise g.Refusal(f"libpq environment variables present ({','.join(bad)}; PG*, KRB5* and GSS* names are refused); "
                        "unset them before any connecting run")


def verify_identity(ctx: Ctx, cur, cls: str, ep: g.Endpoint):
    """V-4/V-9(c): before ANY statement (including SET), connected identity must equal the classified one.
    PLAN-ENV C2: the old local marker-table/token check is gone. A local target needs DISPOSABLE_VERIFIED, which is
    unreachable (owner decision 2026-10-19), so a local connection never gets this far (see disposable_gate)."""
    if cls != "production":
        raise g.Refusal(f"{VERIFY_DISABLED}: no verified-disposable connection exists", g.EXIT_IDENTITY)
    cur.execute("SELECT current_database(), current_user, inet_server_port(), to_regclass('vera_local_meta.disposable_marker') IS NOT NULL")
    db, user, port, has_marker = cur.fetchone()
    if db != ep.dbname or user != ep.role or int(port or 0) != ep.port:
        raise g.Refusal("connected identity does not match the classified target", g.EXIT_IDENTITY)
    if has_marker:
        raise g.Refusal("production target carries a local marker table", g.EXIT_IDENTITY)


def production_path_check(ctx: Ctx):
    """E3p / D-ENV-5: the human production path ships disabled. When enabled it still refuses under an agent marker
    and with ambient PG*/KRB5*/GSS*. An agent can unset markers (L2): the TTY and the typed password are the control."""
    if not PRODUCTION_PATH_ENABLED:
        raise g.Refusal(f"{D_ENV_5_PENDING}: the production migration path is disabled until the owner records a decision")
    present = sorted(k for k in AGENT_MARKERS if ctx.environ.get(k))
    if present:
        raise g.Refusal(f"production migration refused under an agent ({','.join(present)} set); human terminal only")
    check_libpq_env(ctx)


def disposable_gate(ctx: Ctx):
    """E3 / C2: a connecting local op installs the migration connect guard, then needs DISPOSABLE_VERIFIED on the same
    connection. Owner decision 2026-10-19: verification refuses unconditionally, so this always refuses, before any
    connect and before any marker is read. Reasons stay names-only."""
    try:
        ctx.install_guard("migration")
    except Exception as exc:  # noqa: BLE001  fail closed with the same code
        raise g.Refusal(f"{VERIFY_DISABLED} (connect guard unavailable: {type(exc).__name__})", g.EXIT_IDENTITY) from None
    raise g.Refusal(f"{VERIFY_DISABLED}: a local database cannot be verified disposable; connecting refused", g.EXIT_IDENTITY)


def recorded_versions(cur) -> dict[str, tuple[str, str]] | None:
    cur.execute("SELECT to_regclass('vera_vjay.schema_migrations') IS NOT NULL")
    if not cur.fetchone()[0]:
        return None
    cur.execute("SELECT version, sha256, origin FROM vera_vjay.schema_migrations")
    return {v: (s, o) for v, s, o in cur.fetchall()}


def order_checks(batch, recorded: dict | None):
    """V-7(a)/(d) and 'already applied' refusal."""
    if recorded is None:
        if len(batch) != 1 or not g.is_bootstrap(batch[0][1].name, batch[0][2]):   # INC-3 (F-2): name + pinned sha
            raise g.Refusal("tracking table absent: only the bootstrap migration may be applied "
                            f"({g.BOOTSTRAP_NAME} with its pinned reviewed sha256)")
        return
    in_batch = {b[0] for b in batch}
    # K18W (QA F6): until the owner-run baseline (D-5) has written baseline/superseded rows, only the bootstrap is
    # accepted, so historical versions (e.g. 001, whose order check is otherwise vacuous) can never be re-run.
    if not any(o in ("baseline", "superseded") for _, o in recorded.values()):
        if any(not g.is_bootstrap(f.name, data) for _, f, data, _ in batch):   # INC-3 (F-2)
            raise g.Refusal("no baseline rows recorded yet (owner step D-5): only the bootstrap migration may be applied")
    for p, f, _, _ in batch:
        if p in recorded:
            raise g.Refusal(f"{p} is already recorded as applied ({recorded[p][1]}); re-apply refused")
        missing = [f"{n:03d}" for n in range(1, int(p)) if f"{n:03d}" not in recorded and f"{n:03d}" not in in_batch]
        if missing:
            raise g.Refusal(f"{p}: lower versions neither recorded nor superseded: {','.join(missing)}")


# ---------------------------------------------------------------- main
def main(argv: list[str], ctx: Ctx | None = None) -> int:
    ctx = ctx or Ctx()
    try:
        return _main(argv, ctx)
    except g.Refusal as r:
        print(f"REFUSED: {r}", file=ctx.err)
        _audit_safe(ctx, result="refused", code=r.code)
        return r.code
    except Exception as exc:  # noqa: BLE001  r4 (SEC S-6): no traceback, no value or host fragment, fail closed
        print(f"REFUSED: internal error {type(exc).__name__} (details suppressed)", file=ctx.err)
        _audit_safe(ctx, result="refused", code=g.EXIT_REFUSED)
        return g.EXIT_REFUSED
    except KeyboardInterrupt:  # K18W (QA F5): no traceback; any committed list was already printed after connect
        print("INTERRUPTED (details suppressed)", file=ctx.err)
        _audit_safe(ctx, result="interrupted", code=EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED


def _main(argv: list[str], ctx: Ctx) -> int:
    op, opts, ids = parse(argv)
    if op == "rollback":
        return rollback_inspect(ctx, ids[0])
    batch = load_batch(ctx, ids) if ids else []
    cls, ep, prod_fps, seen = classify_targets(ctx)
    if op == "plan":
        check_bindings(opts, batch, required=False)
        print_plan(ctx, op, batch, cls)
        if opts.get("--approve-sha"):
            ctx.say("BINDING every --approve-sha matches the file bytes shown above")
        audit(ctx, op="plan", connected="no", files=len(batch))
        return g.EXIT_REFUSED if any(b[3].hard for b in batch) else g.EXIT_OK
    want = opts["--target"]
    if want == "production":
        production_path_check(ctx)   # E3p: refuses D-ENV-5-pending while PRODUCTION_PATH_ENABLED is False
        ep = prod_endpoint(ctx)
        cls = g.classify(ep, prod_fps | ({ep.fingerprint()} if ep else set()))
    if cls != want:
        raise g.Refusal(f"--target {want} does not match computed class {cls}")
    if cls == "unknown" or ep is None:
        raise g.Refusal("target class unknown; refused")
    if not prod_fps:   # F6: with no fingerprint sources every connecting op is refused (PRODUCTION by construction)
        raise g.Refusal(f"{NO_SOURCES}: no production fingerprint sources found; connecting refused")
    if op == "rehearse" and cls != "local":
        raise g.Refusal("rehearse is local-only")
    if opts.get("--fresh-install") and cls != "local":
        raise g.Refusal("--fresh-install is local-only")
    print_plan(ctx, op, batch, cls)
    # K18W (QA F3/F10): credential and libpq env refusals come first on every connecting path, before any prompt
    check_libpq_env(ctx)
    check_env_credentials(ctx, set() if cls == "production" else g.parse_env_file(ctx.local_env_file)[1])
    if cls == "production":
        if op == "inspect":
            raise g.Refusal("production inspect needs a separately approved read-only route (plan D-3); not offered here")
        oid = production_gate(ctx, op, opts, batch, seen, ep)
        password = ctx.getpass("Admin password (input hidden): ")
        if not password:
            raise g.Refusal("empty password refused (no libpq fallback to .pgpass/PGPASSWORD)")
    else:
        oid = opts.get("--owner-approval", "local")
        check_bindings(opts, batch, required=False)
        for p, f, data, lint in batch:
            hard = [h for h in lint.hard if not (opts.get("--fresh-install") and h in ("DROP SCHEMA", "header says FRESH INSTALL ONLY"))]
            if hard:
                raise g.Refusal(f"{f.name}: hard refusal ({hard[0]})")
            if lint.destructive and not opts.get("--fresh-install") and opts.get("--allow-destructive") != oid:
                raise g.Refusal(f"{f.name}: destructive statements need --allow-destructive {oid}")
        password = ""  # local: peer/socket auth or the local env file's role; never production credentials
    # C3: strip inherited DB/TLS/proxy/admin names AFTER fingerprinting and after the refusal checks above.
    stripped = g.strip_inherited(ctx.environ, STRIP_PATTERNS)
    if stripped:
        ctx.say("STRIPPED inherited names: " + ",".join(stripped))
    if cls != "production":
        disposable_gate(ctx)   # C2: always refuses (VERIFY_DISABLED); the local classification is a routing hint only
    if cls == "production":
        oid = f"{oid}@" + "+".join(g.sha256_hex(b[2])[:12] for b in batch)   # approval_ref records the bound bytes
    try:
        conn = ctx.connect(ep, password, cls)
    except Exception as exc:  # noqa: BLE001  (SEC S-6: never echo the driver message, it names the host)
        print(f"CONNECT FAILED: {type(exc).__name__} (details suppressed: may contain host)", file=ctx.err)
        audit(ctx, op=op, target=cls, result="connect-failed")
        return g.EXIT_CONNECT
    password = ""
    done: list[str] = []   # r5 (W-3): committed versions, reported on any post-connect failure
    in_doubt: list[str] = []   # INC-3 (F-1): a version whose COMMIT was sent but did not return (outcome unknown)
    rc = g.EXIT_REFUSED
    try:
        cur = conn.cursor()
        try:
            verify_identity(ctx, cur, cls, ep)
            if op == "inspect":
                cur.execute("SET TRANSACTION READ ONLY")
                return status(ctx, cur)
            recorded = recorded_versions(cur)
        except g.Refusal:
            raise
        except Exception as exc:  # noqa: BLE001  (SEC S-6)
            print(f"PRE-APPLY CHECK FAILED: {type(exc).__name__} (details suppressed)", file=ctx.err)
            audit(ctx, op=op, target=cls, result="precheck-failed")
            return g.EXIT_CONNECT
        order_checks(batch, recorded)
        conn.rollback()  # end the identity/read transaction before per-file transactions
        for p, f, data, lint in batch:
            try:
                if op == "apply":
                    # INC-3 (WS-8b C1): the write opt-in is the FIRST statement of the per-file transaction (after
                    # every tool-side gate above, before the in-transaction lock/recheck, which take a snapshot).
                    cur.execute(WRITE_OPT_IN)
                cur.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
                cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                cur.execute(f"SELECT pg_advisory_xact_lock({LOCK_KEY})")
                if recorded is not None:
                    cur.execute("SELECT 1 FROM vera_vjay.schema_migrations WHERE version = %s", (p,))
                    if cur.fetchone():
                        raise g.Refusal(f"{p} recorded concurrently; refused")
                cur.execute(data.decode("utf-8"))
                cur.execute("INSERT INTO vera_vjay.schema_migrations (version, filename, sha256, approval_ref, origin) "
                            "VALUES (%s, %s, %s, %s, 'tool')", (p, f.name, g.sha256_hex(data), oid))
                if op == "rehearse":
                    conn.rollback()
                else:
                    in_doubt.append(p)   # INC-3 (F-1): from here until commit() returns the outcome is unknown
                    conn.commit()
                    in_doubt.clear()
                    done.append(p)
            except g.Refusal:
                conn.rollback()
                raise
            except Exception as exc:  # noqa: BLE001
                conn.rollback()   # a failing rollback goes to the outer handler (FAILED (error), in-doubt kept)
                committed = ",".join(done) or "none"
                if in_doubt:   # INC-3 (F-1): COMMIT raised; the server may or may not have committed this file
                    print(f"FAILED on {f.name}: {type(exc).__name__} during COMMIT; OUTCOME UNKNOWN for {f.name} "
                          f"(may or may not be committed; check schema_migrations read-only before any retry). "
                          f"Committed before it: {committed}", file=ctx.err)
                    _audit_safe(ctx, op=op, target=cls, file=f.name, result="outcome-unknown", unknown=f.name,
                                committed=committed)
                else:
                    print(f"FAILED on {f.name}: {type(exc).__name__}; rolled back this file. Committed before it: "
                          f"{committed}", file=ctx.err)
                    _audit_safe(ctx, op=op, target=cls, file=f.name, result="failed", committed=committed)
                return 1
            # INC-3 (F-1): the audit line is written OUTSIDE the per-file except, so an output error after a
            # successful commit is reported by the outer handler as FAILED with the file in the committed list,
            # never as "rolled back this file".
            audit(ctx, op=op, target=cls, file=f.name, sha=g.sha256_hex(data)[:12], approval=oid,
                  result="rolled-back" if op == "rehearse" else "committed")
        rc = g.EXIT_OK
    except g.Refusal as r:
        _safe(conn.rollback)
        if done:   # never let a refusal hide what is already committed
            raise g.Refusal(f"{r} (committed before it: {','.join(done)})", r.code) from None
        raise
    except Exception as exc:  # noqa: BLE001  r5 (W-3): after connect, an unexpected error is FAILED, never REFUSED
        _safe(conn.rollback)
        _post_connect_failed(ctx, op, cls, "error", exc, done, in_doubt)
        rc = 1
    except BaseException as exc:  # K18W (QA F5): Ctrl-C / SystemExit after connect still reports what committed
        _safe(conn.rollback)
        _post_connect_failed(ctx, op, cls, "interrupted", exc, done, in_doubt)
        raise
    finally:
        try:
            conn.close()
        except Exception as exc:  # noqa: BLE001
            _post_connect_failed(ctx, op, cls, "close", exc, done, in_doubt)
            rc = 1
    return rc


def _safe(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001  (a failed rollback is reported by the caller's FAILED line)
        pass


def _audit_safe(ctx: Ctx, **kv):
    """INC-3 (F-1): an audit write that fails (e.g. BrokenPipe on stdout) must not replace the failure record."""
    try:
        audit(ctx, **kv)
    except Exception as exc:  # noqa: BLE001
        print(f"AUDIT-WRITE-FAILED: {type(exc).__name__} (the failure record above stands)", file=ctx.err)


def _post_connect_failed(ctx: Ctx, op: str, cls: str, stage: str, exc: BaseException, done: list[str],
                         in_doubt: list[str] | None = None):
    committed = ",".join(done) or "none"
    unknown = ",".join(in_doubt or []) or "none"
    tail = f"; OUTCOME UNKNOWN for {unknown}" if in_doubt else ""
    print(f"FAILED ({stage}): {type(exc).__name__} (details suppressed). Committed: {committed}{tail}", file=ctx.err)
    _audit_safe(ctx, op=op, target=cls, result="failed", stage=stage, committed=committed, unknown=unknown)


def status(ctx: Ctx, cur) -> int:
    rec = recorded_versions(cur) or {}
    rc = g.EXIT_OK
    names = sorted(e.name for e in os.scandir(ctx.migrations_dir)
                   if g.FILE_RE.fullmatch(e.name) and e.is_file(follow_symlinks=False))   # K18W: no symlinks
    for name in names:
        v = name[:3]
        data = (ctx.migrations_dir / name).read_bytes()
        if v not in rec:
            ctx.say(f"PENDING {name}")
        elif rec[v][1] == "tool" and rec[v][0] != g.sha256_hex(data):
            ctx.say(f"DRIFT {name}")
            rc = g.EXIT_DRIFT
        else:
            ctx.say(f"RECORDED {name} origin={rec[v][1]}")
    for v in sorted(set(rec) - {n[:3] for n in names}):
        ctx.say(f"RECORDED-NO-FILE {v} origin={rec[v][1]}")
    return rc


def rollback_inspect(ctx: Ctx, prefix: str) -> int:
    """HZ-006-RB: the tool never executes a rollback. It shows what the owner-run rollback would destroy."""
    rdir = ctx.migrations_dir / "rollback"
    pat = g.re.compile(prefix + r"_vera_[a-z0-9_]+_rollback\.sql")
    names = []
    if rdir.is_dir():
        with os.scandir(rdir) as it:
            names = [e.name for e in it if pat.fullmatch(e.name) and e.is_file(follow_symlinks=False) and not e.is_symlink()]
    if len(names) != 1:
        raise g.Refusal(f"rollback {prefix}: {'not found' if not names else 'ambiguous'}")
    data = (rdir / names[0]).read_bytes()
    lint = g.lint_sql(names[0], data.decode("utf-8", errors="replace"))
    ctx.say(f"ROLLBACK-FILE {names[0]} sha256={g.sha256_hex(data)[:12]} (inspect only; NOT executed)")
    for d in lint.destructive + lint.advisory:
        ctx.say(f"  WOULD-RUN {d}")
    ctx.say("Rollback is never executed by this tool. Owner-run on an isolated DB only (`psql -1 -f`), after an export,"
            " with separate same-turn approval. If the migration is recorded as applied in production, do not roll back:"
            " use the file's non-destructive alternative.")
    audit(ctx, op="rollback-inspect", connected="no", file=names[0])
    return g.EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

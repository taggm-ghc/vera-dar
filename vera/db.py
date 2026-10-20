import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# DB_URL_ENV_RW is defined once, in mvp_preflight (which must not import this module: no cycle).
from vera.mvp_preflight import DB_URL_ENV_RW

# Render sets RENDER=true in every one of its own build/runtime environments
# (https://render.com/docs/environment-variables). Off-Render (local dev, CI),
# that var is absent, so this falls through to the external URL. This is the
# only place that distinction should be made — everything downstream just
# calls get_engine()/SessionLocal and doesn't know or care which URL backed it.


def _database_url() -> str:
    var = "INTERNAL_DB_URL" if os.getenv("RENDER") == "true" else "EXTERNAL_DB_URL"
    url = os.getenv(var)
    if not url:
        raise RuntimeError(f"{var} is not set (RENDER={os.getenv('RENDER')!r})")
    return url


_engine = None
SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine, SessionLocal
    if _engine is None:
        _engine = create_engine(_database_url(), pool_pre_ping=True, hide_parameters=True)
        SessionLocal = sessionmaker(bind=_engine)
    return _engine


def get_session() -> Session:
    get_engine()
    assert SessionLocal is not None
    return SessionLocal()


def get_role_engine(role: str, *, env=None, engine_factory=create_engine):
    """Engine for one named VERA-DAR account (slim live MVP). Only "rw" exists today; its URL comes from
    the env var named DB_URL_ENV_RW. `engine_factory` is injected by tests so no connection is made.
    The default get_engine() above is unchanged and is NOT used on the MVP run path."""
    if role != "rw":
        raise NotImplementedError(
            f"get_role_engine({role!r}): only the 'rw' role is wired for the slim MVP; wo/ro repointing is "
            "goal #3 (per-role accounts), not built yet.")
    env = os.environ if env is None else env
    url = env.get(DB_URL_ENV_RW)
    if not url:
        raise RuntimeError(f"{DB_URL_ENV_RW} is not set; the rw engine cannot be created "
                           "(the value is never printed).")
    return engine_factory(url, pool_pre_ping=True, hide_parameters=True)

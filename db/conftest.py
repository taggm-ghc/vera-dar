# M3-F (PF-18 M-3): db/ holds operator scripts that read production settings or run against PRODUCTION. They are never
# pytest tests. pytest loads this conftest before it collects any module in this directory (also for an explicit path
# or from another cwd), so the run stops before such a module is imported. NOT effective when --confcutdir or -c puts
# the confcutdir BELOW this directory (route R4c, e.g. --confcutdir=<a subdirectory of it>), under --noconftest (R4), or for a
# direct `python <script>` (R5); see design M3-C/M3-D.
import pytest

raise pytest.UsageError("VERA-DAR: db/ holds operator scripts, not pytest tests; refusing to collect")

"""TABTITLE-1023: every Streamlit page sets a browser-tab title containing "VERA-DAR", and set_page_config is the first
st.* call at module level (Streamlit otherwise raises or falls back to the script file name, e.g. "streamlit_app")."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["streamlit_app.py", "pages/vera_demo.py", "pages/trace_eval.py"]


def _first_st_call(tree: ast.Module):
    calls = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(stmt):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"):
                calls.append(n)
    calls.sort(key=lambda c: (c.lineno, c.col_offset))
    return calls[0] if calls else None


@pytest.mark.parametrize("page", PAGES)
def test_page_title_set_first_and_names_vera_dar(page):
    call = _first_st_call(ast.parse((ROOT / page).read_text()))
    assert call is not None and call.func.attr == "set_page_config", f"{page}: first module-level st.* call must be set_page_config"
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw.get("page_title"), ast.Constant) and "VERA-DAR" in kw["page_title"].value

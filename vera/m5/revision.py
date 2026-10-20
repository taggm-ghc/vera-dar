"""Single bounded revision pass: remove / qualify / tone down flagged claims using only
existing evidence. Never adds new facts; missing evidence is an M4 problem, not a rewrite one."""
from vera.pipeline_llm import UNTRUSTED_NOTE, LLMCallable, call


def revise_response_detailed(response: dict, unsupported_claims: list, *, llm: LLMCallable | None = None) -> dict:
    flagged = "\n".join(f"#{u['claim_index']} [{u['recommended_action']}] {u['claim_text']} -- issues: {'; '.join(u['issues'])}"
                        for u in unsupported_claims)
    system = (
        "Revise the answer to fix flagged claims. For action 'remove': delete the claim. 'qualify': keep it "
        "but state the limitation (contested, weak, outdated or narrow evidence). 'tone_down': weaken it to "
        "what the cited evidence actually shows. Do NOT add any new facts, numbers or citations; keep [E#] "
        "markers on surviving claims. " + UNTRUSTED_NOTE +
        ' Reply JSON: {"revised_text": str, "actions":[{"claim_index":int,"action":"removed|qualified|toned_down",'
        '"new_claim_text": str|null}]}'
    )
    user = (f"<answer>\n{response['response_text']}\n</answer>\nFlagged claims:\n{flagged}")
    res = call(llm, system, user)
    return {"revised_text": str(res.data.get("revised_text", "")).strip(),
            "actions": [a for a in res.data.get("actions", []) if isinstance(a, dict)],
            "model": res.model, "tokens": res.tokens, "cost_usd": res.cost_usd, "latency_s": res.latency_s}


def revise_response(response: dict, unsupported_claims: list, *, llm: LLMCallable | None = None) -> str:
    return revise_response_detailed(response, unsupported_claims, llm=llm)["revised_text"]

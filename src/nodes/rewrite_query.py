"""Rewrite-query node: rephrases the current query using the LLM, increments retries."""

import httpx

_PROMPT = """\
A RAG system tried to answer the following question but the answer quality was low.
Rephrase the question to be more specific so that better context can be retrieved.
Respond with ONLY the rephrased question — no explanation, no preamble.

Original question: {query}
Quality issue: {reason}
Rephrased question:"""


async def rewrite_query(state: dict, *, llm_url: str, llm_model: str) -> dict:
    reason = "low faithfulness or relevance"
    if state.get("eval_result") and state["eval_result"].judge_reasoning:
        reason = state["eval_result"].judge_reasoning

    prompt = _PROMPT.format(query=state["query"], reason=reason)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{llm_url}/api/generate",
            json={"model": llm_model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        new_query = resp.json()["response"].strip()

    return {
        "query": new_query,
        "retries": state.get("retries", 0) + 1,
    }

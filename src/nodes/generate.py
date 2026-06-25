"""Generate node: builds a grounded prompt and calls Ollama /api/generate."""

import httpx


async def generate(state: dict, *, llm_url: str, llm_model: str) -> dict:
    context = "\n\n".join(f"[Source: {c.source_filename}]\n{c.text}" for c in state["chunks"])
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't have that information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\nAnswer:"
    )
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{llm_url}/api/generate",
            json={"model": llm_model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        answer = resp.json()["response"]
    return {"answer": answer, "eval_result": None}

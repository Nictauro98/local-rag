"""Evaluate node: runs the composite evaluator and stores EvalResult in state."""

from src.evaluation.interfaces.evaluator import Evaluator


async def evaluate(state: dict, *, evaluator: Evaluator) -> dict:
    context = [c.text for c in state["chunks"]]
    result = await evaluator.evaluate(state["query"], context, state["answer"])
    result = result.model_copy(update={"retries": state.get("retries", 0)})
    return {"eval_result": result}

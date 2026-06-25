"""
AWS Bedrock embedder adapter — migration stub.

Migration steps:
1. Add boto3 to dependencies: `uv add boto3`.
2. Set env vars: RAG_AWS_REGION, RAG_BEDROCK_EMBED_MODEL (e.g. amazon.titan-embed-text-v2:0).
3. Replace with boto3.client("bedrock-runtime", region_name=...) and wrap in asyncio.to_thread.
4. embed    → bedrock.invoke_model(modelId=model, body=json.dumps({"inputText": text}))
              parse response: json.loads(body)["embedding"]
5. dimension → invoke once with a dummy string; return len(embedding).
6. Switch RAG_EMBEDDER_BACKEND=aws in .env.
"""

from src.interfaces.embedder import Embedder


class BedrockEmbedder(Embedder):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("BedrockEmbedder is a migration stub. See module docstring.")

    async def embed_one(self, text: str) -> list[float]:
        raise NotImplementedError("BedrockEmbedder is a migration stub. See module docstring.")

    def dimension(self) -> int:
        raise NotImplementedError("BedrockEmbedder is a migration stub. See module docstring.")

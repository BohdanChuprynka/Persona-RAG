from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persona_rag.index.embedder import embed_batch


@pytest.mark.asyncio
async def test_embed_batch_calls_openai_once() -> None:
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    with patch("persona_rag.index.embedder._client", return_value=fake_client):
        vecs = await embed_batch(["a", "b", "c"])
    assert len(vecs) == 3
    assert len(vecs[0]) == 1536
    fake_client.embeddings.create.assert_called_once()

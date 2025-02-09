import pytest
from sheldor.models import Document, OllamaEmbedding, OllamaLLM, InMemoryVectorStore
from sheldor.main import Sheldor
from sheldor.exceptions import SheldorError


@pytest.fixture
async def rag_system():
    return await Sheldor()


@pytest.mark.asyncio
async def test_add_document(rag_system):
    content = "Test document content"
    metadata = {"source": "test", "type": "unit_test"}

    await rag_system.add_document(content, metadata)
    assert len(rag_system.vector_store.documents) == 1
    assert rag_system.vector_store.documents[0].content == content
    assert rag_system.vector_store.documents[0].metadata == metadata


@pytest.mark.asyncio
async def test_query(rag_system):
    # Add test document
    await rag_system.add_document(
        "Python is a programming language", {"source": "test"}
    )

    # Test query
    response = await rag_system.query("What is Python?")
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_empty_query(rag_system):
    response = await rag_system.query("What is Python?")
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_personality_traits(rag_system):
    """Test if responses contain Sheldon's personality traits."""
    response = await rag_system.query("What is quantum mechanics?")

    # Check for characteristic Sheldon traits
    assert any(
        [
            "theoretical physicist" in response,
            "physics" in response.lower(),
            "scientific" in response.lower(),
            "precisely" in response.lower(),
            "actually" in response.lower(),  # Sheldon often uses this to correct others
        ]
    ), "Response should contain Sheldon's characteristic language"


@pytest.mark.asyncio
async def test_scientific_accuracy(rag_system):
    """Test if responses maintain scientific accuracy."""
    await rag_system.add_document(
        content="Quantum mechanics describes nature at atomic scales.",
        metadata={"source": "physics_textbook"},
    )

    response = await rag_system.query("Explain quantum mechanics")
    assert "quantum" in response.lower(), "Response should contain scientific content"
    assert (
        "mechanics" in response.lower()
    ), "Response should maintain scientific accuracy"

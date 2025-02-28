import pytest
from unittest.mock import AsyncMock, patch
from sheldor.models import Document, OllamaEmbedding, OllamaLLM, InMemoryVectorStore
from sheldor.main import Sheldor
from sheldor.exceptions import SheldorError
from sheldor.ui import StreamlitUI
import tempfile
from pathlib import Path


@pytest.fixture
async def rag_system():
    return await Sheldor()


@pytest.fixture
def mock_uploaded_file():
    """Fixture to create a mock uploaded PDF file."""

    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content

        def getvalue(self):
            return self.content

    return MockFile("test.pdf", b"Sample PDF content")


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


@pytest.mark.asyncio
@patch("sheldor.ui.st.session_state.rag_system", new_callable=AsyncMock)
async def test_process_uploaded_file(mock_rag_system, mock_uploaded_file):
    """Test the processing of an uploaded PDF file."""
    ui = StreamlitUI()

    # Mock the add_document method to simulate adding a document
    mock_rag_system.add_document = AsyncMock()

    # Call the process_uploaded_file method
    await ui.process_uploaded_file(mock_uploaded_file)

    # Check if the add_document method was called with the correct parameters
    assert mock_rag_system.add_document.called
    assert mock_rag_system.add_document.call_count == 1
    args, kwargs = mock_rag_system.add_document.call_args[0]
    assert kwargs["metadata"]["source"] == mock_uploaded_file.name
    assert kwargs["metadata"]["type"] == "pdf"
    assert (
        kwargs["content"] == "Sample PDF content"
    )  # Adjust based on actual content extraction logic

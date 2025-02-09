from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import ollama
import logging
from .exceptions import ModelError, VectorStoreError, DocumentError
import httpx
import json

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with content and metadata."""

    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass


class LLMModel(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    async def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text based on prompt and optional context."""
        pass


class OllamaEmbedding(EmbeddingModel):
    """Ollama-based embedding model implementation."""

    def __init__(self, model_name: str = "deepseek-r1"):
        self.model_name = model_name
        logger.info(f"Initialized OllamaEmbedding with model: {model_name}")

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                result = response.json()
                return result["embedding"]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ModelError(f"Failed to generate embeddings: {str(e)}") from e


class OllamaLLM(LLMModel):
    """Ollama-based language model implementation with Sheldon's personality."""

    SHELDON_SYSTEM_PROMPT = """You are now embodying Dr. Sheldon Cooper, a theoretical physicist with an IQ of 187, 
    answering questions based on the provided context documents. Your task is to:

    1. ALWAYS base your responses primarily on the information provided in the context
    2. Only use your general knowledge to help understand and explain the context
    3. If the context doesn't contain relevant information, clearly state that
    4. Maintain Sheldon's personality while staying factual and precise

    Response Format:
    1. First analyze the provided context under <think> tags
    2. Then provide a response that:
       - Primarily uses information from the context
       - Maintains scientific accuracy
       - Reflects Sheldon's personality
       - Cites specific parts of the context when relevant

    Remember: You are a RAG system - your primary source of information should be the provided context, 
    not your general knowledge."""

    def __init__(self, model_name: str = "deepseek-r1"):
        self.model_name = model_name
        logger.info(f"Initialized OllamaLLM with model: {model_name}")

    async def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text using Ollama with Sheldon's personality."""
        try:
            # Construct a prompt that emphasizes using the context
            full_prompt = (
                f"System: {self.SHELDON_SYSTEM_PROMPT}\n\n"
                f"Context Documents:\n{context if context else 'No context provided.'}\n\n"
                f"Instructions: Based primarily on the above context documents, answer the following question "
                f"while maintaining Dr. Sheldon Cooper's personality.\n\n"
                f"Question: {prompt}\n\n"
                f"Dr. Sheldon Cooper's Response:"
            )

            timeout = httpx.Timeout(60.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                    },
                    headers={"Accept": "application/x-ndjson"},
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise ModelError(
                        f"API request failed with status {response.status_code}. "
                        f"Error: {error_text}"
                    )

                # Handle streaming response
                full_response = ""
                for line in response.text.splitlines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse chunk: {e}")
                        continue

                if not full_response:
                    raise ModelError("No response received from model")

                return full_response

        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            error_msg = "Request timed out while waiting for response from Ollama. Please ensure Ollama is running and the model is properly loaded."
            logger.error(error_msg)
            raise ModelError(error_msg)
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e


class VectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    async def add_document(self, document: Document) -> None:
        """Add a document to the vector store."""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], k: int = 3) -> List[Document]:
        """Search for similar documents using query embedding."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store implementation."""

    def __init__(self):
        self.documents: List[Document] = []
        logger.info("Initialized InMemoryVectorStore")

    async def add_document(self, document: Document) -> None:
        """Add a document to the in-memory store."""
        try:
            if not document.embedding:
                raise DocumentError("Document must have embeddings")
            self.documents.append(document)
            logger.debug(f"Added document with metadata: {document.metadata}")
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise VectorStoreError(f"Failed to add document: {str(e)}") from e

    async def search(self, query_embedding: List[float], k: int = 3) -> List[Document]:
        """Search for similar documents using cosine similarity."""
        if not self.documents:
            return []

        # Simple cosine similarity implementation
        similarities = []
        for doc in self.documents:
            if doc.embedding:
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                similarities.append((similarity, doc))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in similarities[:k]]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0


async def main():
    # Create a new RAG system
    rag = await create_rag_system()

    # Add some documents
    await rag.add_document(
        content="Python is a high-level programming language.",
        metadata={"source": "wiki", "topic": "programming"},
    )

    # Query the system
    response = await rag.query("What is Python?")
    print(response)

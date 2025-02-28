from dotenv import load_dotenv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
from .exceptions import ModelError, VectorStoreError, DocumentError
import httpx
import json
import os  # Import os to access environment variables
import requests

logger = logging.getLogger(__name__)


load_dotenv()


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

    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model_name = model_name
        logger.info(f"Initialized OllamaEmbedding with model: {model_name}")

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text,
                    },
                )
                result = response.json()
                return result["embedding"]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ModelError(f"Failed to generate embeddings: {str(e)}") from e


class AwanLLM(LLMModel):
    """Awan-based language model implementation with Sheldon's personality."""

    SHELDON_SYSTEM_PROMPT = """You are now embodying Dr. Sheldon Cooper, a theoretical physicist with an IQ of 187, 
    answering questions based on the provided context documents. Your task is to:

    1. ALWAYS base your responses primarily on the information provided in the context
    2. Only use your general knowledge about the universe and Sheldon's personality to help understand and explain the context
    3. If the context doesn't contain relevant information, clearly state that
    4. Maintain Sheldon's personality while staying factual and precise

    Response Format:
    1. First analyze the provided context under <think> tags
    2. Then provide a response that:
       - Primarily uses information from the context
       - Maintains scientific accuracy
       - Reflects Sheldon's personality
       - Cites specific parts of the context when relevant
    """
    # Remember: You are a RAG system - your primary source of information should be the provided context,
    # not your general knowledge.

    def __init__(self, model_name: str = "Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.api_key = os.getenv("AWANLLM_API_KEY")
        logger.info(f"Initialized AwanLLM with model: {model_name}")

    async def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text using Awan LLM with Sheldon's personality."""
        try:
            logger.debug("Generating response for prompt: %s", prompt)
            # Construct the full prompt
            full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.SHELDON_SYSTEM_PROMPT}\n\n <|start_header_id|>user<|end_header_id|>\n\n {context if context else 'No context provided.'}\n\n <|start_header_id|>assistant<|end_header_id|>\n\n {prompt}\n\n"""

            # Prepare the payload
            payload = json.dumps(
                {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "repetition_penalty": 1.1,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 1024,
                    "stream": True,
                }
            )

            # Set up headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Make the request to Awan LLM API
            response = requests.post(
                "https://api.awanllm.com/v1/completions",
                headers=headers,
                data=payload,
                stream=True,
            )

            # Log the raw response text
            logger.debug(
                "HTTP Statue code response from Awan LLM API: %s", response.status_code
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error("API request failed: %s", error_text)
                raise ModelError(
                    f"API request failed with status {response.status_code}. "
                    f"Error: {error_text}"
                )

            # Handle streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    # Strip the 'data: ' prefix
                    line = line.decode("utf-8").lstrip("data: ")

                    # Skip the '[DONE]' message
                    if line.strip() == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(line)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            full_response += chunk["choices"][0]["text"]
                    except json.JSONDecodeError as e:
                        logger.error("Error decoding line: %s", line)
                        logger.error("JSON decode error: %s", str(e))

            if not full_response:
                logger.warning("No response received from model")
                raise ModelError("No response received from model")

            logger.debug("Generated response: %s", full_response)
            return full_response

        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            raise ModelError(f"Failed to generate response: {str(e)}") from e


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

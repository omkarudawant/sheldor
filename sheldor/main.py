from typing import Optional
from .models import (
    Document,
    EmbeddingModel,
    LLMModel,
    VectorStore,
    OllamaEmbedding,
    AwanLLM,
    InMemoryVectorStore,
)
import logging
from .exceptions import SheldorError  # , ModelError
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# import httpx
# import requests
# import json
# import os  # Import os to access environment variables
import streamlit as st
import asyncio
import tempfile
from pathlib import Path
import fitz
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

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


class Sheldor:
    """Main RAG system implementation."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        llm_model: Optional[LLMModel] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize Sheldor with its components."""
        self.embedding_model = embedding_model or OllamaEmbedding()
        self.llm_model = llm_model or AwanLLM()
        self.vector_store = vector_store or InMemoryVectorStore()
        logger.info("Initialized Sheldor RAG system")

    async def add_document(self, content: str, metadata: dict) -> None:
        """Add a document to the RAG system."""
        try:
            embedding = await self.embedding_model.embed(content)
            document = Document(
                content=content,
                metadata=metadata,
                embedding=embedding,
            )
            await self.vector_store.add_document(document)
            logger.info(f"Successfully added document with metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise SheldorError(f"Failed to add document: {str(e)}") from e

    async def query(
        self, question: str, additional_context: Optional[str] = None
    ) -> str:
        """Query the RAG system with a question."""
        try:
            # Generate embeddings for the question
            query_embedding = await self.embedding_model.embed(question)

            # Search for relevant documents
            relevant_docs = await self.vector_store.search(query_embedding, k=3)

            if not relevant_docs:
                context = "No relevant documents found in the knowledge base."
            else:
                # Format document contents with metadata
                doc_contexts = []
                for doc in relevant_docs:
                    doc_context = (
                        f"[Source: {doc.metadata['source']}, "
                        f"Page: {doc.metadata['page']}, "
                        f"Chunk: {doc.metadata['chunk']}]\n{doc.content}"
                    )
                    doc_contexts.append(doc_context)

                context = "\n\n---\n\n".join(doc_contexts)

            # Combine with additional context if provided
            if additional_context:
                full_context = f"{additional_context}\n\n{context}"
            else:
                full_context = context

            # Log the full context being sent to the LLM
            logger.debug("Full context being sent to LLM: %s", full_context)

            # Generate response
            response = await self.llm_model.generate(
                question,
                context=full_context,
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise SheldorError(f"Failed to process query: {str(e)}") from e


async def create_rag_system(
    embedding_model_name: str = "mxbai-embed-large",
    llm_model_name: str = "Meta-Llama-3-8B-Instruct",
) -> Sheldor:
    """Factory function to create a new Sheldor instance."""
    embedding_model = OllamaEmbedding(model_name=embedding_model_name)
    llm_model = AwanLLM(model_name=llm_model_name)
    vector_store = InMemoryVectorStore()

    return Sheldor(
        embedding_model=embedding_model,
        llm_model=llm_model,
        vector_store=vector_store,
    )


async def process_query(self, query: str) -> None:
    """Process user query and update chat history."""
    # Add the user's query to the session state before processing
    st.session_state.messages.append({"role": "user", "content": query})

    try:
        with st.spinner("Processing with the precision of a theoretical physicist..."):
            # Directly await the async function
            response = await self._process_query(query)

            # Add assistant's response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Force a rerun to update the UI
            st.rerun()

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(str(e))  # Display the error message


async def process_uploaded_file(self, uploaded_file) -> None:
    """Process the uploaded PDF file."""
    try:
        with st.spinner(
            "Processing document with Sheldon's meticulous attention to detail..."
        ):
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            # Process PDF synchronously
            pdf_document = fitz.open(tmp_path)
            try:
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text.strip():
                        # Directly await the add_document method
                        await st.session_state.rag_system.add_document(
                            content=text,
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_num + 1,
                                "chunk": f"chunk_{page_num + 1}",
                                "type": "pdf",
                            },
                        )
            finally:
                pdf_document.close()  # Ensure the PDF document is closed

            st.success(f"Successfully processed {uploaded_file.name}!")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(str(e))


async def handle_query(query: str) -> None:
    """Handle the user query and process it."""
    await process_query(query)  # Call the process_query function


async def main():
    """Main function to run the Streamlit app."""
    # existing code...

    if query := st.chat_input(
        "Ask a question (I promise to be more patient than with Penny)"
    ):
        await handle_query(query)  # Call the new async function


if __name__ == "__main__":
    asyncio.run(main())  # Use asyncio.run to execute the main function

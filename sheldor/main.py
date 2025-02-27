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
from .exceptions import SheldorError

logger = logging.getLogger(__name__)


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
    embedding_model_name: str = "llama2",
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

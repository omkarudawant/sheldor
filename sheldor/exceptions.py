class SheldorError(Exception):
    """Base exception class for Sheldor."""

    def __str__(self):
        return f"As a theoretical physicist, I must point out this error: {super().__str__()}"


class ModelError(SheldorError):
    """Raised when there's an error with LLM or embedding models."""

    def __str__(self):
        return (
            f"Even with my IQ of 187, I encountered a model error: {super().__str__()}"
        )


class VectorStoreError(SheldorError):
    """Raised when there's an error with vector storage operations."""

    def __str__(self):
        return f"Good Lord! A vector store error: {super().__str__()}"


class DocumentError(SheldorError):
    """Raised when there's an error processing documents."""

    def __str__(self):
        return f"This document is more troublesome than Penny's acting career: {super().__str__()}"

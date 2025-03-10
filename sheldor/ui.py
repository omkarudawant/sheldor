import asyncio
import streamlit as st
import tempfile
import filetype
from pathlib import Path
import logging
import ollama
import fitz  # PyMuPDF

from sheldor.main import create_rag_system
from sheldor.document_processor import PDFProcessor
from sheldor.logging_config import setup_logging

logger = logging.getLogger(__name__)


class StreamlitUI:
    """Streamlit UI for the Sheldor RAG system."""

    def __init__(self):
        setup_logging()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "rag_system" not in st.session_state:
            # Initialize the RAG system properly
            st.session_state.rag_system = None  # Set to None initially
            asyncio.create_task(self.create_rag_system())  # Schedule the task

    async def create_rag_system(self):
        """Create the RAG system asynchronously."""
        st.session_state.rag_system = await create_rag_system(
            embedding_model_name="mxbai-embed-large",
            llm_model_name="Meta-Llama-3-8B-Instruct",
        )

    async def process_uploaded_file(self, uploaded_file) -> None:
        """Process the uploaded PDF file."""
        try:
            with st.spinner(
                "Processing document with Sheldon's meticulous attention to detail..."
            ):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)

                # Process PDF synchronously
                pdf_document = fitz.open(tmp_path)
                try:
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        text = page.get_text()
                        if text.strip():
                            # Schedule the add_document method to run asynchronously
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
                    print("PDF document closed.")

                st.success(f"Successfully processed {uploaded_file.name}!")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(str(e))

    async def _process_query(self, query: str) -> str:
        """Process query asynchronously."""
        try:
            response = await st.session_state.rag_system.query(query)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def process_query(self, query: str) -> None:
        """Process user query and update chat history."""
        try:
            with st.spinner(
                "Processing with the precision of a theoretical physicist..."
            ):
                # Directly await the async function
                response = await self._process_query(query)

                # Add messages to session state
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Force a rerun to update the UI
                st.rerun()

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            st.error(str(e))

    def display_message(self, message: dict) -> None:
        """Display a message with special formatting for thinking sections."""
        with st.chat_message(message["role"]):
            content = message["content"]

            # Check for thinking section
            if "<think>" in content and "</think>" in content:
                # Split into thinking and response
                parts = content.split("</think>")
                thinking = parts[0].replace("<think>", "").strip()
                response = parts[1].strip() if len(parts) > 1 else ""

                # Display thinking in darker grey box with improved typography
                st.markdown(
                    """
                    <div style="
                        background-color: #2F3336;
                        color: #E1E1E1;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 15px;
                        font-family: 'Courier New', monospace;
                        font-size: 0.9em;
                        line-height: 1.4;
                    ">
                        <div style="
                            color: #9CA3AF;
                            font-family: system-ui, -apple-system, sans-serif;
                            font-size: 0.85em;
                            margin-bottom: 8px;
                            font-style: italic;
                        ">
                            💭 Thinking Process
                        </div>
                        {}
                    </div>
                """.format(
                        thinking
                    ),
                    unsafe_allow_html=True,
                )

                # Display main response
                if response:
                    st.markdown(response)
            else:
                # Regular message without thinking section
                st.markdown(content)

    async def render(self):
        """Render the Streamlit UI."""
        st.title("Sheldor - The Sheldon Cooper RAG System")
        st.markdown(
            """
        *"I'm not crazy, my mother had me tested."* - Dr. Sheldon Cooper
        
        Welcome to Sheldor, a RAG system that combines the intelligence of a theoretical physicist 
        with an IQ of 187 and the personality of Dr. Sheldon Cooper.
        """
        )

        # File upload section
        st.subheader("Document Upload")
        st.markdown(
            """
        As I always say, proper documentation is the cornerstone of scientific inquiry. 
        Please upload your PDF documents for analysis.
        """
        )

        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded_file:
            # Check if file is PDF using filetype
            file_bytes = uploaded_file.getvalue()
            kind = filetype.guess(file_bytes)

            if kind and kind.mime == "application/pdf":
                if st.button("Process Document"):
                    await self.process_uploaded_file(uploaded_file)
            else:
                st.error(
                    "I must insist that you upload a proper PDF file. As my mother always says, 'If you want to do something, do it right.'"
                )

        # Check if rag_system is initialized
        if st.session_state.rag_system is None:
            st.warning("Initializing the RAG system, please wait...")
            return  # Exit early if rag_system is not ready

        # Chat interface
        st.divider()
        st.subheader("Engage in Intellectual Discourse")

        # Display chat messages
        for message in st.session_state.messages:
            self.display_message(message)

        # Query input
        if query := st.chat_input(
            "Ask a question (I promise to be more patient than with Penny)"
        ):
            await self.process_query(query)


async def main():
    ui = StreamlitUI()
    await ui.render()  # Use await here instead of asyncio.run


if __name__ == "__main__":
    main()

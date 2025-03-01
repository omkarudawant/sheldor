# Sheldor - Scalable RAG System
> "I'm not crazy, my mother had me tested." - Dr. Sheldon Cooper

Sheldor is a sophisticated Retrieval Augmented Generation (RAG) system that combines the intelligence of a theoretical physicist
with an IQ of 187 and the personality of Dr. Sheldon Cooper. Using local LLMs and embeddings via Ollama, it provides
precise, scientifically accurate responses with a touch of Sheldon's unique personality.

## Features

- **Sheldon Cooper's Personality**: Responses embody Sheldon's characteristic traits and communication style
- **Local LLM Integration**: Uses Awan LLM for chat completion
- **PDF Document Processing**: Support for uploading and processing PDF documents
- **Streamlit UI**: Clean and intuitive chat interface
- **Modular Architecture**: Easy to extend and modify components
- **Async Support**: Built with asyncio for better performance
- **Error Handling**: Comprehensive error handling and logging
- **Type Hints**: Full type annotation support
- **Testing**: Unit tests for critical components

## Prerequisites

As Sheldon would say, "Proper preparation prevents poor performance."

1. Install [Ollama](https://ollama.ai/download)
2. Obtain an API key from Awan LLM and set it as an environment variable:
```bash
export AWANLLM_API_KEY='your_api_key_here'
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sheldor.git
cd sheldor
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit UI:
```bash
streamlit run app.py
```

2. Open your browser and navigate to http://localhost:8501

3. Use the interface to:
   - Upload PDF documents
   - Chat with the RAG system about the documents
   - View conversation history

## Project Structure

```
sheldor/
├── __init__.py
├── main.py              # Core RAG implementation
├── models.py            # Abstract classes and implementations
├── document_processor.py # PDF processing
├── ui.py               # Streamlit interface
├── exceptions.py       # Custom exceptions
├── logging_config.py   # Logging setup
└── config.py           # Configuration management

tests/
└── test_sheldor.py     # Unit tests
```

## Development

1. Run tests:
```bash
pytest tests/
```

2. Check code formatting:
```bash
black sheldor/
```

3. Run type checking:
```bash
mypy sheldor/
```

## Configuration

The system can be configured through environment variables:
- `SHELDOR_DEFAULT_LLM_MODEL`: Default LLM model (default: "Meta-Llama-3-8B-Instruct")
- `SHELDOR_DEFAULT_EMBEDDING_MODEL`: Default embedding model (default: "deepseek-r1")
- `SHELDOR_LOG_LEVEL`: Logging level (default: "INFO")
- `SHELDOR_SARCASM_DETECTION`: Enable sarcasm detection (default: False)
- `SHELDOR_VERBOSITY`: Control explanation detail level (default: 3)
- `SHELDOR_SCIENTIFIC_REFERENCES`: Include scientific paper references (default: True)
- `AWANLLM_API_KEY`: API key for Awan LLM service

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Awan LLM](https://awanllm.com/) for chat completion support
- [Streamlit](https://streamlit.io/) for the UI framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

## Docker Usage

### Building the Docker Image

To build the Docker image for Sheldor, run the following command in the root directory of the project:

```bash
docker build -t sheldor .
```

### Running the Docker Container

After building the image, you can run the container with the following command:

```bash
docker run -p 8501:8501 sheldor
```

This will start the Streamlit application, and you can access it at `http://localhost:8501`.

### Prerequisites

As Sheldon would say, "Proper preparation prevents poor performance."

1. Install [Ollama](https://ollama.ai/download)
2. Obtain an API key from Awan LLM and set it as an environment variable:
```bash
export AWANLLM_API_KEY='your_api_key_here'
```

---
```



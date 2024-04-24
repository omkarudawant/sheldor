# Sheldor
---

Sheldor is a chatbot project inspired by the character Sheldon Lee Cooper from the TV show "The Big Bang Theory". It leverages FastAPI to provide an interactive conversational experience with a touch of Sheldon's personality.

## Features

- **Ollama**: Get up and running with local LLMs.
- **Langchain**: 🦜🔗 Build context-aware reasoning applications.
- **FastAPI Backend**: Utilizes FastAPI for creating a fast and modern API backend.
- **Python 3.10+**: Built using Python, ensuring compatibility with the latest language features.
- **Poetry for Dependency Management**: Manages project dependencies with Poetry for streamlined development.
- **Modular Structure**: Organized codebase with separate modules for routes, models, and tests.
- **Easy Installation**: Quick setup with Poetry for dependency installation and virtual environment management.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/omkarudawant/sheldor.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sheldor
    ```

3. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

## Usage

1. Start the FastAPI server:

    ```bash
    poetry run uvicorn sheldor.main:app --reload
    ```

2. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs) to interact with Sheldor.

## Contributing

Contributions are welcome! If you'd like to contribute to Sheldor, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Create a new pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) - FastAPI framework for building APIs with Python.
- [The Big Bang Theory](https://www.cbs.com/shows/big_bang_theory/) - Inspiration for the character Sheldon Lee Cooper.

---

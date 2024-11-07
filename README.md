# byoLLM
Bring your own LLM, local UI for your purposes.

This Project is targeted to allow peopl


# Local Llama

This tool lets you interact with your PDFs, TXT, or Docx files & LLMs entirely offline, removing the need for OpenAI or cloud dependencies. It uses local Language Models for improved privacy and fully offline capabilities.


## Features

- Full Offline Capability: Works even without an internet connection
- Local LLM Support: Uses Ollama for high-efficiency local processing
- Versatile File Support: Handles PDF, TXT, DOCX, and MD files
- Persistent Document Indexing: Stores and reuses embedded documents
- User-Friendly Interface: Accessible via Streamlit


## New Updates

- Ollama integration for significant performance improvements
- Uses `nomic-embed-text` (Ollama Default) and `llama3.2:1b` models (models are customizable)
    - ref `text_embedder.py`:
    ```
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434",
    ```
- Upgraded to Haystack 2.0 for improved processing
- Integrated a persistent Chroma vector database for reusing uploaded documents


## Setup Instructions

1. Download Ollama from [here](https://ollama.ai/download)
2. Clone this repository
3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Load the necessary Ollama models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ```

## Running the Application

1. Start the Ollama server:
   ```
   ollama serve
   ```
2. Launch the Streamlit application:
   ```
   python -m streamlit run LLM_Chat_Interface.py
   ```
3. Upload your documents and start chatting!



## How It Works

1. Document Indexing: Uploaded files are processed, split, and embedded using Ollama.
2. Vector Storage: Embeddings are stored in a local Chroma vector database.
3. Query Matching: User inputs are embedded, and related document sections are identified and retrieved.
4. Response Generation: Ollama generates responses based on the retrieved context and chat history.

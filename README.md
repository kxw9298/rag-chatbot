# RAG Chatbot with LLM and Vector Database

This repository contains the code and setup for a Retrieval-Augmented Generation (RAG) chatbot that leverages a large language model (LLM) and a PostgreSQL vector database for semantic search. The chatbot is built with Docker, uses Streamlit for the UI, and includes Jupyter notebooks for testing database interactions.

## Repository Structure

```plaintext
.
├── chatbot
│   ├── Dockerfile               # Dockerfile to build the chatbot container
│   ├── chatbot_logic.py         # Core chatbot logic, including interaction with LLM and vector database
│   ├── chatbot_ui.py            # Streamlit-based UI for the chatbot
│   └── requirements.txt         # Python dependencies for the chatbot
├── docker-compose.yml            # Docker Compose configuration for the entire project
├── document_processor
│   ├── Dockerfile               # Dockerfile to build the document processor container
│   ├── app.py                   # Flask app to manage document processing
│   ├── process_documents.py      # Logic for processing and storing document embeddings in the vector database
│   └── requirements.txt         # Python dependencies for the document processor
├── documents
│   └── carbon-free-energy.pdf    # Sample PDF document to be processed for embeddings
├── logs                          # Directory for storing logs
└── notebooks
    ├── Dockerfile               # Dockerfile to build the Jupyter Notebook container
    ├── requirements.txt         # Python dependencies for the notebook
    └── vector-database.ipynb     # Jupyter notebook for manual interaction with the vector database
```

## Purpose of the Repo

The goal of this project is to create a chatbot that can retrieve relevant information from internal documents stored in a PostgreSQL vector database, pass the context to an LLM (e.g., OpenAI GPT), and generate meaningful responses. The core components of the system include:

- **LLM for natural language processing and generation**: This project uses OpenAI's API for question answering.
- **RAG architecture**: The chatbot retrieves document embeddings from the vector database using semantic search and provides relevant context to the LLM to generate more accurate responses.
- **PostgreSQL vector database**: PGVector is used to store and retrieve embeddings of the documents.
- **Streamlit**: A web-based interface for users to interact with the chatbot.

## Prerequisites

Before you proceed with setting up the project, ensure you have the following:

- **Docker** and **Docker Compose** installed
- Access to an **OpenAI API Key** (for embedding generation and LLM interactions)

### Step 1: Get Your OpenAI API Key

To get your OpenAI API key, follow these steps:

1. Sign up at [OpenAI](https://platform.openai.com/signup).
2. Go to your account and navigate to the API keys section.
3. Create an API key and copy it.

### Step 2: Configure Environment Variables

Create a `.env` file in the project root and add the following environment variables:

```plaintext
# PostgreSQL Database Configuration
POSTGRES_HOST=localhost
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=vector_db

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Collection Name for PGVector
COLLECTION_NAME=documents
```

## Building and Running the Containers

### Step 3: Build and Run with Docker Compose

To build and run the containers for the chatbot, document processor, and Jupyter Notebook, follow these steps:

1. Open a terminal and navigate to the project directory.
2. Run the following command to build and start the services using Docker Compose:

```bash
docker-compose up --build
```

This will build and run the following services:

- **PostgreSQL**: Vector database for storing document embeddings.
- **Document Processor**: Processes PDF documents and stores their embeddings in the PostgreSQL database.
- **Chatbot**: The Streamlit-based chatbot UI for interacting with the system.
- **Jupyter Notebook**: A notebook for interacting with the vector database and testing the embeddings.

### Step 4: Access the Jupyter Notebook

The Jupyter Notebook is useful for manually interacting with the PostgreSQL vector database and embeddings. To access the notebook:

1. Open your browser and navigate to:
   ```
   http://localhost:8888
   ```
2. The main notebook is located at:
   ```
   notebooks/vector-database.ipynb
   ```

Use this notebook to test the embeddings stored in the database and query them for similarities.

### Step 5: Access the Chatbot UI

The chatbot interface is built using Streamlit, allowing users to interact with the system via a simple web-based chat UI. To access the chatbot:

1. In your browser, go to:
   ```
   http://localhost:8501
   ```
2. Enter your questions in the chat, and the chatbot will retrieve relevant document embeddings from the vector database and generate answers using the LLM.

### Step 6: Process Documents

To process new documents and add their embeddings to the vector database, follow these steps:

1. Place the PDF files you want to process in the `documents/` directory.
2. Run the document processor script (inside the Docker container or locally):

```bash
python document_processor/process_documents.py --file-path /path/to/document.pdf
```

This script will split the document into chunks, generate embeddings, and store them in the PostgreSQL vector database.

## Conclusion

This project demonstrates the use of a chatbot with Retrieval-Augmented Generation (RAG), using a PostgreSQL vector database and an LLM to answer questions based on internal document data. It provides a flexible solution for document-based question answering using semantic search and advanced NLP capabilities.

## References

- [OpenAI API](https://platform.openai.com/)
- [LangChain](https://github.com/hwchase17/langchain)
- [PostgreSQL PGVector](https://github.com/pgvector/pgvector)
- [Streamlit](https://streamlit.io/)
- [Docker](https://www.docker.com/)

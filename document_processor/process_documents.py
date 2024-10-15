import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Retrieve necessary environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')  # Default to 'documents' if not set

def process_and_store_document(file_path):
    """
    Processes the document, splits it into chunks, generates embeddings,
    and stores them in the PostgreSQL vector database using PGVector.from_documents().
    """
    try:
        # Load and split the PDF document
        logging.info(f"Loading and splitting the document: {file_path}")
        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = loader.load_and_split(text_splitter)
        logging.info(f"Successfully split the document into {len(documents)} chunks")

        # Initialize OpenAI embeddings model
        logging.info("Initializing OpenAI embeddings model")
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Connection string for PostgreSQL
        connection_string = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}'

        # Clean document content and remove null characters
        logging.info("Cleaning document content")
        for document in documents:
            document.page_content = document.page_content.replace('\x00', '')  # Clean null characters

        # Use PGVector.from_documents to embed and store documents in the vector database
        logging.info(f"Storing {len(documents)} document chunks in the vector database")
        db = PGVector.from_documents(
            embedding=embedding_model,
            documents=documents,
            collection_name=COLLECTION_NAME,
            connection_string=connection_string,
            use_jsonb=True  # Optional, depending on your use case
        )

        logging.info(f"Document processing and storage complete: {file_path}")
        return len(documents)
    
    except Exception as e:
        logging.error(f"Error occurred during document processing: {e}")
        return None

if __name__ == "__main__":
    file_path = "/path/to/your/document.pdf"  # Example file path; should be passed in real scenarios
    num_chunks = process_and_store_document(file_path)

    if num_chunks:
        logging.info(f"Successfully processed and stored {num_chunks} document chunks.")
    else:
        logging.error("Failed to process and store the document.")

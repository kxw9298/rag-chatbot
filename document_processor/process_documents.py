import os
from dotenv import load_dotenv  # New import
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings 

# Load environment variables from .env file
load_dotenv()

POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def process_and_store_document(file_path):
    """
    Processes the document, splits it into chunks, generates embeddings,
    and stores them in the PostgreSQL vector database.
    """
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = loader.load_and_split(text_splitter)

    # Initialize OpenAI embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create vector search interface to interact with the PostgreSQL database
    vector_search = PGVector(
        collection_name='documents', 
        connection_string=f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}',
        embedding_function=embedding_model,
        use_jsonb=True
    )

    # Process and clean document content
    for document in documents:
        document.page_content = document.page_content.replace('\x00', '')

    # Store the processed documents in the PostgreSQL vector database
    vector_search.add_documents(documents)
    
    return len(documents)

import os
import psycopg2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def connect_to_vector_db():
    """
    Connect to the PostgreSQL vector database.
    """
    return psycopg2.connect(
        host=POSTGRES_HOST,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB
    )

def perform_semantic_search(query_text, top_k=2):
    """
    Perform a semantic search on the vector database to retrieve the most relevant context.
    """
    # Initialize the OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create the vector search interface
    vector_search = PGVector(
        collection_name='documents',
        connection_string=f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}',
        embedding_function=embeddings
    )
    
    # Perform the semantic search on the user's query
    results = vector_search.similarity_search(query_text, k=top_k)
    
    # Extract and format the context
    context = "\n".join([result.page_content for result in results])
    
    return context

def query_llm(query, context):
    """
    Send the user query and the retrieved context to the LLM (OpenAI GPT).
    """
    # Initialize the LLM model
    llm = OpenAI(api_key=OPENAI_API_KEY)
    
    # Combine the query and context for the LLM prompt
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    # Get the response from the LLM
    response = llm(prompt)
    
    return response

def handle_user_query(query):
    """
    Handle the user's query by retrieving relevant context from the vector database
    and querying the LLM with the context.
    """
    # Perform semantic search to get the relevant context
    context = perform_semantic_search(query)
    
    if not context:
        return "Sorry, I couldn't find any relevant information."
    
    # Send the query and context to the LLM
    response = query_llm(query, context)
    
    return response

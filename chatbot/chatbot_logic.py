import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "vector_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI for embeddings and LLM
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Set up the PGVector connection
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=POSTGRES_HOST,
    port=5432,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

vector_search = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model,
)

# Memory for conversation history
memory = ConversationBufferWindowMemory(
    memory_key="history",
    ai_prefix="Bot",
    human_prefix="User",
    k=3,  # Store the last 3 conversation pairs
)

def search_similar_documents(query):
    """
    Perform a similarity search using PGVector.
    """
    found_docs = vector_search.similarity_search(query)
    return "\n\n".join([doc.page_content for doc in found_docs])

def generate_prompt(query, context, history):
    """
    Build the full prompt for the LLM based on the current query, context, and conversation history.
    """
    prompt = f"""
    You are a helpful assistant who finds answers based on the provided context and the conversation history.

    text_context:
    {context}

    conversation_history:
    {history}

    query:
    {query}
    """
    return prompt

def get_response_from_llm(query, context, history):
    """
    Get the final response from the OpenAI LLM, based on the prompt.
    """
    prompt = generate_prompt(query, context, history)
    response = llm(prompt)
    return response

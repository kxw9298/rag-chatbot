import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "vector_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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

def build_chat_history(history):
    """
    Convert conversation history into a list of messages for the new OpenAI API.
    """
    messages = []
    if history:
        for message in history["history"]:
            messages.append({"role": message["role"], "content": message["content"]})
    return messages

def get_response_from_llm(query, context, history):
    """
    Get the final response from the OpenAI LLM using the new API structure.
    """
    # Prepare messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
    ]

    # Append the conversation history if available
    messages.extend(build_chat_history(history))

    # Call the OpenAI API (new API structure for v1.0.0+)
    response = openai.completions.create(
        model="gpt-3.5-turbo",  # or 'gpt-4' if available
        messages=messages,
        max_tokens=500
    )

    # Extract and return the response from the assistant
    return response.choices[0].message.content

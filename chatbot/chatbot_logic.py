import os
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate

# Load environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "vector_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Set up the PGVector connection
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=POSTGRES_HOST,
    port=5432,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
)
COLLECTION_NAME = os.getenv("POSTGRES_DB", "documents")

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

## Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who helps in finding answers using the provided context."),
    ("human", """
        The answer should be based on the text context given in "text_context" and the conversation history given in "conversation_history" along with its Caption: \n
        Base your response on the provided text context and the current conversation history to answer the query.
        Select the most relevant information from the context.
        Generate a draft response using the selected information. Remove duplicate content from the draft response.
        Generate your final response after adjusting it to increase accuracy and relevance.
        Now only show your final response!
        If you do not know the answer or context is not relevant, respond with "I don't know".

        text_context:
        {context}

        conversation_history:
        {history}

        query:
        {query}
    """)
])

def search_similar_documents(query):
    """
    Perform a similarity search using PGVector. If no documents are found, return None.
    """
    found_docs = vector_search.similarity_search(query)
    if not found_docs:
        print("not found similar documents")
        return None
    return "\n\n".join([doc.page_content for doc in found_docs])

def build_chat_history(history):
    """
    Convert conversation history into a list of messages for the ChatCompletion API.
    Ensure that history is a list of dictionaries with 'role' and 'content' keys.
    """
    messages = []
    # Check if history is a list of dictionaries
    if isinstance(history, list):
        for message in history:
            # Ensure each message is a dictionary with the required keys
            if isinstance(message, dict) and "role" in message and "content" in message:
                messages.append({"role": message["role"], "content": message["content"]})
            else:
                print(f"Invalid message format: {message}")
    else:
        print("History is not in the expected format (list of dictionaries)")

    return messages


def get_response_from_llm(query, context, history):
    """
    Get the final response from the OpenAI LLM using the correct API interface.
    If no relevant context is found, return a fallback message.
    """
    # Return fallback response if context is missing
    if not context and not history:
        return "I don't know, it is outside of my knowledge."

    # Prepare messages for the chat completion API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context if context else 'No relevant context found'}\n\nQuery: {query}"}
    ]

    # Append the conversation history if available
    messages.extend(build_chat_history(history))

    chat_completion = client.chat.completions.create(
    messages=messages,
    model="gpt-3.5-turbo",
    max_tokens=500
    )

    # Call the OpenAI API with the new `completions` method
    response = chat_completion

    # Extract and return the response from the assistant
    return response.choices[0].message.content

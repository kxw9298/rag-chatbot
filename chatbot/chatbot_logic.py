import os
import logging
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "vector_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
logging.debug("Initializing OpenAI embeddings")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)

# Set up the PGVector connection
logging.debug("Setting up PGVector connection")
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
logging.debug("Setting up conversation memory")
memory = ConversationBufferWindowMemory(
    memory_key="history",
    ai_prefix="Bot",
    human_prefix="User",
    k=3,  # Store the last 3 conversation pairs
)

# Define the prompt template
logging.debug("Defining prompt template")
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
    Perform a similarity search using PGVector. Log the documents and their similarity scores for debugging purposes.
    """
    logging.debug(f"Searching for documents similar to: {query}")

    # Perform similarity search with scores
    found_docs = vector_search.similarity_search_with_score(query)  # Returns documents and scores

    if not found_docs:
        logging.info("No similar documents found.")
        return None

    # Log the found documents and their similarity scores
    for doc, score in found_docs:
        logging.debug(f"Document: {doc.page_content[:100]}... | Similarity Score: {score}")

    # Set a threshold for similarity
    # threshold = 0.6  # You can adjust this threshold based on your dataset
    # relevant_docs = [doc for doc, score in found_docs if score > threshold]

    # if not relevant_docs:
    #     logging.info(f"No documents found above the similarity threshold of {threshold}.")
    #     return None

    # Return the combined content of all found documents
    return "\n\n".join([doc.page_content for doc, _ in found_docs])

def build_chat_history(history):
    """
    Convert conversation history into a list of messages for the ChatCompletion API.
    Ensure that history is a list of dictionaries with 'role' and 'content' keys.
    """
    messages = []
    # Check if history is a list of dictionaries
    if isinstance(history, list):
        for message in history:
            if isinstance(message, dict) and "role" in message and "content" in message:
                messages.append({"role": message["role"], "content": message["content"]})
            else:
                logging.warning(f"Invalid message format: {message}")
    else:
        logging.warning("History is not in the expected format (list of dictionaries)")

    return messages

def get_response_from_llm(query, context, history):
    """
    Get the final response from the OpenAI LLM using the correct API interface.
    If no relevant context is found, return a fallback message.
    """
    logging.debug("Getting response from LLM")

    # If no relevant context is found, return fallback response
    if not context:
        logging.info("No relevant context found. Returning fallback response.")
        return "I don't know, the provided context does not mention anything about your question."

    # Build the conversation messages for the LLM using the prompt template
    formatted_prompt = prompt_template.format_messages(
        name="Bot",
        context=context,
        history=history,
        query=query
    )

    # Ensure all messages include a 'role' and 'content' field
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context if context else 'No relevant context found'}\n\nQuery: {query}"}
    ]

    # Append history if available
    if history:
        messages.extend(build_chat_history(history))

    # Log the messages being sent to the LLM
    logging.debug(f"Messages for LLM: {messages}")

    # Call the OpenAI API
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=500
        )
        logging.debug("LLM response received")
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting response from LLM: {e}")
        return "There was an error processing your request."


def handle_query(user_input, history):
    """
    Handle the user's query by searching for similar documents and getting a response from the LLM.
    """
    logging.debug(f"Handling user query: {user_input}")

    # Search for similar documents based on the user's query
    context = search_similar_documents(user_input)

    # If no similar documents, return "I don't know"
    if not context:
        logging.info("No relevant documents found in embeddings. Returning fallback response.")
        return "I don't know, the provided context does not mention anything about your question."

    # Get the response from the LLM
    response = get_response_from_llm(user_input, context, history)

    logging.debug(f"LLM response: {response}")
    return response

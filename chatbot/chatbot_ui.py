import streamlit as st
from chatbot_logic import handle_query
from langchain.memory import ConversationBufferWindowMemory

# Initialize Streamlit UI
st.title("🤖 Chatbot with OpenAI and Vector Search")

# Memory to store conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "ai", "content": "How can I help you today?"}]

# Memory to store previous context
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(
        memory_key="history",
        ai_prefix="Bot",
        human_prefix="User",
        k=3,  # Store the last 3 conversation exchanges
    )

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input via chat interface
if chat_input := st.chat_input("Ask a question"):
    with st.chat_message("human"):
        st.write(chat_input)
        st.session_state.messages.append({"role": "human", "content": chat_input})

    # Load history from memory
    history = st.session_state.memory.load_memory_variables({})

    # Call handle_query to process the user input and history
    response = handle_query(chat_input, history)

    # Display the AI response
    with st.chat_message("ai"):
        st.write(response)
        st.session_state.messages.append({"role": "ai", "content": response})

    # Save the conversation history
    st.session_state.memory.save_context({"input": chat_input}, {"output": response})

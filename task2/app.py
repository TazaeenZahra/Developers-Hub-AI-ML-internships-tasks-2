import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
# ---------- CONFIG ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- LOAD VECTOR DB ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

# ---------- LOAD LLM ----------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# ---------- UI ----------
st.title("📚 MTH RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_query = st.chat_input("Ask a question...")

if user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

     # ===== MEMORY PART =====
        chat_history = ""

    for msg in st.session_state.messages:
        chat_history += f"{msg['role']}: {msg['content']}\n"

    # Retrieve relevant chunks

    search_query = f"""
    Conversation History:
    {chat_history}

    Current Question:
    {user_query}
    """
    
    docs = vectordb.similarity_search(search_query, k=2)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Use the conversation history and retrieved context to answer.

Rules:
- Give direct answers.
- Do not say "according to the context".
- Do not say "based on the retrieved information".
- If the answer is not available in the context, say:
  "I couldn't find that information in the knowledge base."
- Keep answers concise and relevant.

Conversation History:
{chat_history}

Retrieved Context:
{context}

Current Question:
{user_query}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)
# Context-Aware Chatbot using RAG and LangChain

## Overview

This project is a Context-Aware Chatbot built using LangChain, ChromaDB, HuggingFace Embeddings, Groq LLM, and Streamlit.

The chatbot retrieves information from a custom knowledge base and maintains conversational context using chat history.

## Features

* Retrieval-Augmented Generation (RAG)
* ChromaDB Vector Database
* HuggingFace Embeddings
* Groq Large Language Model
* Context-Aware Conversations
* Streamlit User Interface


## Workflow

1. Load the knowledge base.
2. Split text into chunks.
3. Generate embeddings.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks based on user queries.
6. Pass retrieved context and conversation history to Groq.
7. Generate responses and display them through Streamlit.


## Create Vector Database

```bash
python create_vector_db.py
```

## Run Application

```bash
streamlit run app.py
```

## Technologies Used

* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq
* Streamlit

## Author

Tazaeen Zahra

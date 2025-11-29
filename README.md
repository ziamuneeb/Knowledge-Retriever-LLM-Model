# KnowledgeRetriever LLM Model

1. Project Title

KnowledgeRetriever – A Retrieval-Augmented LLM that answers questions from your notes or documents.

2. Description

KnowledgeRetriever is a Python-based tool that leverages LangChain, vector embeddings (FAISS), and GPT LLM Model to answer questions from text data such as lecture notes, documentation, or research papers. It splits your text into chunks, stores them in a vector database, and performs retrieval-augmented generation (RAG) to provide accurate answers.

3. Features

Split text into manageable chunks using RecursiveCharacterTextSplitter.

Store and search text chunks using FAISS vector database.

Supports local LLMs (e.g., GPT4All) or cloud LLMs (OpenAI, Hugging Face).

Retrieve answers to user queries with context from your notes.

Easily extendable for multiple documents or datasets.

4. Installation

Step-by-step instructions:

# Clone the repo
git clone link


# Create a virtual environment
python -m venv venv
# Activate it
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

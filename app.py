from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load notes
with open("data/ML.txt", "r") as f:
    text = f.read()

# Retrieving API Key as a environmnetal variable
API_KEY = os.environ.get("OPENAI_API_KEY")


# Step 2: Split notes into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
docs = splitter.split_text(text)

# Step 3: Create vector database
# Vector DB
embeddings = OpenAIEmbeddings(api_key=API_KEY)
vectorstore = FAISS.from_texts(docs, embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever()

# Step 5: Build LCEL Retrieval Pipeline
prompt = PromptTemplate(
    template="""You are a helpful assistant.

Use ONLY the following context to answer the question:

{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini")

# LCEL pipeline
chain = (
    {
        "context": retriever,
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6: Ask a question
query = "What are the types of machine learning?"
response = chain.invoke({"question": query})

print("Q:", query)
print("A:", response)


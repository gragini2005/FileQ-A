# FileQ-A
This project demonstrates a File Question Answering System using LangChain and Open Source Large Language Models (LLMs). The system allows users to upload documents (PDF, Text, etc.) and ask questions based on the content. The AI extracts information, understands context, and provides accurate answers.

Objectives Build an intelligent File Q&A system Use LangChain for document processing and retrieval Implement embeddings and vector databases Use open-source LLMs for query answering Enable natural language interaction with uploaded files

Technologies Used Python LangChain SentenceTransformers / HuggingFace Embeddings FAISS / ChromaDB PyPDF2 / TextLoader OpenAI / LLaMA / GPT4All models Google Colab / Jupyter Notebook

System Workflow 1️ Upload file (PDF, TXT, DOCX) 2️ Extract text from file 3️ Convert text chunks into embeddings 4️ Store embeddings in a vector database 5️ User asks a question in natural language 6️ System retrieves relevant context 7️ LLM generates an answer from file content

Features ✔ Upload and process different file formats ✔ Semantic similarity search using embeddings ✔ Context-aware answers ✔ Supports open-source & closed-source models ✔ Interactive chatbot interface (optional)

Example Use Cases Use Case Description Education Students upload lecture notes and ask questions Business Reports Analyze and query company reports Legal Documents Search and query contracts and policies Research Papers Extract answers from academic articles

Future Enhancements 🔹 Streamlit / Flask-based Web App 🔹 Support for multiple file uploads 🔹 Chat history and memory integration 🔹 API deployment using FastAPI / Docker

Conclusion This project shows how AI can transform documents into interactive knowledge bases using LangChain, embeddings, and open-source LLMs. It is a practical introduction to Retrieval-Augmented Generation (RAG) systems

!pip install langchain langchain-community
!pip install faiss-cpu
!pip install pypdf python-docx
!pip install sentence-transformers
!pip install transformers
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pick loader based on file type
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".docx"):
    loader = Docx2txtLoader(file_path)
else:
    loader = TextLoader(file_path)

docs = loader.load()

# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

print(f"Total Chunks: {len(documents)}")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Small + Fast embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents, embeddings)

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Load small model
flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base", #base means small  # change to flan-t5-small for even faster
    max_length=512 #character or token
)

llm = HuggingFacePipeline(pipeline=flan_pipeline)

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# Test
query = "Give me a short summary of the document"
print(qa.run(query))

while True:
    q = input("Ask a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print("Answer:", qa.run(q))

https://colab.research.google.com/drive/11YOLySFXDz9c2GIj2o4TTTOqZ80DWZ9W?usp=sharing

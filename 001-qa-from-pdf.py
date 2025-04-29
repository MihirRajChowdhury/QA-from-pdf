# import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())
# groq_api_key = os.environ["GROQ_API_KEY"]

# from langchain_groq import ChatGroq

# # Create LLM instance using Groq API
# llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# from langchain_community.document_loaders import PyPDFLoader

# file_path = "./data/Be_Good.pdf"

# # Load PDF document
# loader = PyPDFLoader(file_path)
# docs = loader.load()

# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Import Hugging Face embeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# # Create Hugging Face embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Split the documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# # Create a vector store with the embeddings
# vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

# # Create a retriever from the vector store
# retriever = vectorstore.as_retriever()

# # Import from langchain_core instead of langchain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_core.prompts import ChatPromptTemplate

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# # Create the question-answering chain
# question_answer_chain = create_stuff_documents_chain(llm, prompt)

# # Create the RAG chain
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # Invoke the chain with a query
# response = rag_chain.invoke({"input": "What is this article about?"})

# print("\n----------\n")
# print("What is this article about?")
# print("\n----------\n")
# print(response["answer"])
# print("\n----------\n")

# print("\n----------\n")
# print("Show metadata:")
# print("\n----------\n")
# print(response["context"][0].metadata)
# print("\n----------\n")

import os 
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

groq_api_key = os.environ["GROQ_API_KEY"]

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

from langchain_community.document_loaders import PyPDFLoader

file_path = "./data/Be_Good.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(docs[0].page_content[0:100])

print(docs[0].metadata)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits,embedding=embedding_model)
retriver = vectorstore.as_retriever()

# from langchain.chains import create_retrival_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate(
    [
        ("system",system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)

rag_chain = create_retrieval_chain(retriver,question_answer_chain)

response = rag_chain.invoke({"input": "what is this article about"})

print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

print("\n----------\n")
print("Show metadata:")
print("\n----------\n")
print(response["context"][0].metadata)
print("\n----------\n")
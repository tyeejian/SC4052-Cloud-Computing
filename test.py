from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import torch

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = pipeline("text-generation" ,
    model_name,
    torch_dtype="auto",
    device_map="auto")


llm = HuggingFacePipeline(pipeline=model)
prompt = """
    You are a professional documentation writer. You can write in-depth documentation or README based on code. 
    
    Your task is to refurbished exisiting README when given one, write README of a given code, or write a summary of a given code. Follow the rules below.

    1. Skip reading the line "Search: " in the first line.
    2. A new README should include minimally
        - Note saying this is generated from LLM (Italics)
        - Table of contents
        - About
        - How to build
        - Documentation
        - License
        - Contacts
        - Technology stack used (if applicable)

    Include in your own knowledge what is needed for a README. Use the context below

    {context}
"""

prompt_template = PromptTemplate.from_template(prompt)


loader = GitLoader(
    clone_url="https://github.com/jack-thant/civic-otters",
    repo_path="./repo/test_repo/",
    branch="main",
)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=20
)

texts = text_splitter.split_documents(data)

db = Chroma.from_documents(texts, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
retriever = db.as_retriever()


chain = {"context" : retriever} | prompt_template | llm

response = chain.invoke()

print(response)
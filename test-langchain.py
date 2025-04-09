from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma

import requests
import base64

loader = GitLoader(
    clone_url="https://github.com/jack-thant/civic-otters",
    repo_path="./repo/test_repo/",
    branch="master",
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

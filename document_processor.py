from dotenv import load_dotenv
from pydantic import Field

#IMPORTANT: load environment variables before import so that langchain modules can access it
load_dotenv()

from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mistralai import MistralAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores.base import VectorStore
from langchain_ollama import ChatOllama


from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch

import os


class DocumentEncoders:
    def __init__(self, file_dir: str = "./documents",
                 embeddings: Embeddings = MistralAIEmbeddings(model="mistral-embed"),
                 model: BaseChatModel = init_chat_model("mistral-large-latest", model_provider="mistralai")):
        self._file_dir = file_dir
        self.embeddings = embeddings
        self._model = model
        self.vector_store = self.get_vector_store()

    def encode_all(self):
        file_names = os.listdir(self._file_dir)
        for name in file_names:
            self.encode(name)

    def encode(self, file_name: str):
        doc = self.load_doc(file_name)
        splits = self.split_doc(doc)
        self.store_doc(splits)

    def load_doc(self, file_name: str) -> list[Document]:
        file_path = os.path.join(self._file_dir, file_name)
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        print(f"Loaded {len(doc)} pages from {file_name}...")
        return doc

    def split_doc(self, docs: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        return all_splits

    def store_doc(self, docs: List[Document]):
        _ = self.vector_store.add_documents(documents=docs)

    def get_vector_store(self) -> VectorStore:
        client = MongoClient(
            os.getenv("MONGODB_URI")
        )
        collection = client[os.getenv("MONGODB_DATABASE")][os.getenv("MONGODB_COLLECTION")]
        return MongoDBAtlasVectorSearch(
            embedding=self.embeddings,
            collection=collection,
            index_name=os.getenv("MONGODB_VECTOR_INDEX"),
            relevance_score_fn="cosine"
        )



encoder = DocumentEncoders()

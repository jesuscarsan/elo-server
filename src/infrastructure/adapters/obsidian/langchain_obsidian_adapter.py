import os
from typing import List
import logging
import sys

# Suppress Chroma telemetry and other noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Global patch for posthog to silence the telemetry error
# The error "capture() takes 1 positional argument but 3 were given" suggests 
# a mismatch in the installed posthog library.
try:
    import posthog
    def noop_capture(*args, **kwargs):
        pass
    posthog.capture = noop_capture
except ImportError:
    pass

from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.config import Settings
from src.domain.ports.obsidian_port import ObsidianPort

class LangChainObsidianAdapter(ObsidianPort):
    def __init__(self, vault_path: str, google_api_key: str, persist_directory: str = "../../workspace/chromadb"):
        self.vault_path = vault_path
        self.persist_directory = persist_directory
        
        # 1. Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
            task_type="retrieval_document"
        )
        
        # 2. Initialize LLM for the chain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=google_api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        self.vector_store = self._initialize_vector_store()
        self.rag_chain = self._create_rag_chain()

    def _initialize_vector_store(self):
        # Check if DB already exists
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    client_settings=Settings(anonymized_telemetry=False)
                )
            except Exception:
                # Fallback if there's any issue with the persist directory
                pass
        
        # Otherwise, load and index the vault
        loader = ObsidianLoader(self.vault_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        return Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            client_settings=Settings(anonymized_telemetry=False)
        )

    def _create_rag_chain(self):
        retriever = self.vector_store.as_retriever()
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    def query(self, question: str) -> str:
        try:
            response = self.rag_chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            return f"Error executing query: {str(e)}"

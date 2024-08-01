import logging
import sys
import os
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import datetime

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-turbo",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01",
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01",
)

from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model

def load_documents(directory):
    return SimpleDirectoryReader(directory).load_data()


def build_index(documents):
    return VectorStoreIndex.from_documents(documents)

def query_index(index, query):
    query_engine = index.as_query_engine()
    return query_engine.query(query)

if __name__ == "__main__":
    try:
        pdf_path = "C:/Users/HTR036/Downloads/LAB TASK"
        document_name = os.path.basename(pdf_path)
        last_update_date = datetime.datetime.fromtimestamp(os.path.getmtime(pdf_path)).isoformat()
       
        documents = load_documents(pdf_path)
        index = build_index(documents)
        response = query_index(index, "What is attention?")
        print(response)
    except Exception as e:
        logging.error(f"Error in main script: {e}")
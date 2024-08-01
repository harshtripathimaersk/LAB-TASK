# Import necessary libraries
import logging
import os
from dotenv import load_dotenv
import sys
import textwrap
import pickle
from redis import Redis
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import StorageContext


# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Initialize Azure OpenAI and embedding models
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


# Update settings
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model

index_key = "what did you learn about the author?"
# load documents
def load_documents(directory):
    return SimpleDirectoryReader(directory).load_data()

# Function to build index
def build_index(documents):
    return VectorStoreIndex.from_documents(documents)

# Create Redis client connection
redis_client = Redis.from_url("redis://localhost:6379")

# Create the vector store wrapper
vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)

# Load storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Check if index exists in Redis
index_exists = redis_client.exists(index_key)

if index_exists:
    # Load index from Redis
    logging.info("Loading index from Redis...")
    dict_bytes = redis_client.get(index_key)
    index = pickle.loads(dict_bytes)
    
else:
    # Create index from documents
    logging.info("Creating index from documents...")
    documents = load_documents("./data/paul_graham")
    index = build_index(documents)
    dict_bytes = pickle.dumps(index)
    # Save index to Redis
    redis_client.append(index_key,dict_bytes)

# Create query engine and retriever
query_engine = index.as_query_engine()
retriever = index.as_retriever()

# some code to demo the retrieval process
result_nodes = retriever.retrieve("What did the author learn?")
for node in result_nodes:
    print(node)
print(redis_client.keys())
# Query the index
response = query_engine.query("What did the author learn?")
print("Response: ",textwrap.fill(str(response), 100))
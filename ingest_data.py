from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

#API Key
openai_api_key = ""

# Load Data
loader = UnstructuredFileLoader("state_of_the_union.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

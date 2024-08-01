import dotenv
from langchain.document_loaders.directory import DirectoryLoader 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

BOOKS_FILE_PATH ="/home/vscode/books/"
BOOKS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = DirectoryLoader(BOOKS_FILE_PATH, glob="**/*.pdf")
reviews = loader.load_and_split()
review_vector_db = Chroma.from_documents(
    documents=reviews,
    embedding=OpenAIEmbeddings(),
    persist_directory=BOOKS_CHROMA_PATH,   
)
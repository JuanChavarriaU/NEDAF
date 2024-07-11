from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from ViewModel import config 
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)




#load docs from the specified directory
loader = DirectoryLoader(config.FILE_DIR, glob="*.pdf")

documents = loader.load() #getting an empty list
# split the text to chuncks of of size 1000
text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
    is_separator_regex = True,
)
# Split the documents into chunks of size 1000 using a CharacterTextSplitter object
texts = text_splitter.split_documents(documents)

# Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)

def answer(prompt: str, persist_directory: str = config.PERSIST_DIR) -> str:
    LOGGER.info(f"Start answering based on prompt: {prompt}")

    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    doc_chain = load_qa_chain(
        llm=OpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0,
            max_tokens=300,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )

    LOGGER.info(f"The top {config.k} chunks are consider to answer the user's query")

    qa = RetrievalQA(
        retriever=docsearch.as_retriever(),
        combine_documents_chain=doc_chain,
    )

    result = qa.run({"query":prompt})
    answer = result["result"]

    LOGGER.info(f"The returned answer is: {answer}")

    LOGGER.info(f"Answering module over")
    return answer
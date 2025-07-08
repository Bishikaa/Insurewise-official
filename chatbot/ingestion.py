from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# loading
def load_all_docs():
    file1 = TextLoader('/Users/bishikachhetri/Desktop/official project/txtfile/1stsuryajyoti.txt').load()
    file2 = TextLoader('/Users/bishikachhetri/Desktop/official project/txtfile/2ndlicnepal.txt').load()
    file3 = TextLoader('/Users/bishikachhetri/Desktop/official project/txtfile/3rdsanima.txt').load()
    file4  = TextLoader('/Users/bishikachhetri/Desktop/official project/txtfile/4thmettlife.txt').load()
    file5 = TextLoader('/Users/bishikachhetri/Desktop/official project/txtfile/5thsunlife.txt').load()
    return file1,file2,file3,file4,file5

doc1, doc2, doc3, doc4, doc5= load_all_docs()


# splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 250,
    separators= ['\n\n','\n','.',' ','']
)
splitted1 = splitter.split_documents(doc1)
splitted2 = splitter.split_documents(doc2)
splitted3 = splitter.split_documents(doc3)
splitted4 = splitter.split_documents(doc4)
splitted5 = splitter.split_documents(doc5)


def vectorestore(collectionname,directory,documents):
    vectore_store = Chroma(
        collection_name= collectionname,
        embedding_function= embeddings,
        persist_directory= directory
    )
    vectore_store.add_documents(documents)
    return vectore_store

vs1 = vectorestore('insurance1.vdb', 'insurance1.db', splitted1)
vs2 = vectorestore('insurance2.vdb', 'insurance2.db', splitted2)
vs3 = vectorestore('insurance3.vdb', 'insurance3.db', splitted3)
vs4 = vectorestore('insurance4.vdb', 'insurance4.db', splitted4)
vs5 = vectorestore('insurance5.vdb', 'insurance5.db', splitted5)



from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config.settings import OPENAI_EMBEDDING_MODEL

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

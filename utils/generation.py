from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from config.settings import MEMORY_KEY, OPENAI_MODEL

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model=OPENAI_MODEL)
    memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

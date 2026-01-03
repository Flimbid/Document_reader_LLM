import os
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import ConversationalRetrievalChain


API_KEY ="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"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,max_tokens=500, openai_api_key=API_KEY)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

vector_db = Chroma(collection_name="custom_gpt_collection", embedding_function=embeddings, persist_directory="./chroma_db")

retriever =  ContextualCompressionRetriever(
    base_retriever=vector_db.as_retriever(),
    base_compressor=LLMChainExtractor.from_llm(llm)
)

prompt_template = ChatPromptTemplate.from_template("""
Context: {context}
Chat History: {chat_history}
Human: {question}
AI: Please provide a relevant answer based on the context and chat history.

""")

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

def chat_response(user_message):
    return chain.invoke({"question": user_message})["answer"]
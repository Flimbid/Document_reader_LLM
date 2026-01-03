from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
import os


API_KEY ="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"

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

vector_db = Chroma(collection_name="custom_gpt_collection", embedding_function=embeddings, persist_directory="./chroma_db")

def add_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    
    vector_db.add_documents(split_docs)

    print("Documents added to the vector database.")


def main():
    while True:
        file_path = input("Enter the path of the text file to add to the vector database (or 'exit' to quit): ")
        
        if file_path.lower() == 'exit':
            break
        if file_path.lower() == 'q':
            break
        if os.path.exists(file_path):
            add_documents(file_path)
        else:
            print("Invalid file path. Please try again.")

    
if __name__ == "__main__":
    main()
    
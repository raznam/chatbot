# embed_documents.py

def embed_documents():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    # Load PDF files from the directory
    loader = DirectoryLoader(
        path="./data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
        show_progress=True
    )
    docs = loader.load()
    #print(docs)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    #print(chunks)

    # Use HuggingFace embeddings sentence-transformers/all-MiniLM-L12-v2
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    # Store embeddings in Chroma
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    vectorstore.add_documents(chunks)
    #vectorstore.persist()

    print("✅ Documents embedded and saved to ChromaDB.")
    print("Documents in ChromaDB:", vectorstore._collection.count())

if __name__ == "__main__":
    embed_documents()

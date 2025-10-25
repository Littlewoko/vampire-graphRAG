from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from typing import List

def get_raw_docs(search_query: str):
    loader = WikipediaLoader(
        query=search_query,
        load_max_docs=10,
        lang="en",
    )
    docs = loader.load()
    return docs

def chunk_docs(docs: List):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    chunks = text_splitter.split_documents(docs)
    return chunks
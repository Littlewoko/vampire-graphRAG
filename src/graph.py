from typing import List
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

def get_graph_representation(chunks: List, llm: ChatOpenAI):
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)
    return graph_documents

def store_graph_documents(graph_documents: List, knowledge_graph: Neo4jGraph):
    knowledge_graph.add_graph_documents(
        graph_documents,
        include_source=True,
        baseEntityLabel=True
    )

def generate_knowledge_graph(search_query: str, llm: ChatOpenAI, knowledge_graph: Neo4jGraph):
    from docs import get_raw_docs, chunk_docs
    raw_docs = get_raw_docs(search_query)
    chunks = chunk_docs(raw_docs)
    graph_documents = get_graph_representation(chunks, llm)
    store_graph_documents(graph_documents, knowledge_graph)
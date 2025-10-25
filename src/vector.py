from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings

def get_vector_index():
    return Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
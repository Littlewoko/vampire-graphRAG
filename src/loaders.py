from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

def load_knowledge_graph():
    knowledge_graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    return knowledge_graph

def load_llm():
    LLM = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")
    return LLM
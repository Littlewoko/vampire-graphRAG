from langchain_neo4j import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from vector import get_vector_index
from entities import Entities

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = ""
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str, entity_chain, knowledge_graph: Neo4jGraph) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = knowledge_graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response])
    return result

def retriever(question: str, entity_chain, knowledge_graph: Neo4jGraph):
    vector_index = get_vector_index()
    structured_data = structured_retriever(question, entity_chain, knowledge_graph)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""
    Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ".join(unstructured_data)}
    """
    return final_data
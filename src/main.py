from loaders import load_llm, load_knowledge_graph
from graph import generate_knowledge_graph

from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Any, Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

def get_vector_index():
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
    return vector_index

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str, entity_chain: Any, knowledge_graph: Neo4jGraph) -> str:
    """
    Collects the neighbourhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print(f" Getting Entity: {entity}")
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
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        # print(response)
        result += "\n".join([el["output"] for el in response])
    return result

def retriever(question: str, entity_chain: Any, knowledge_graph: Neo4jGraph):
    vector_index = get_vector_index()
    print(f"Search query: {question}")
    structured_data = structured_retriever(question, entity_chain, knowledge_graph)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""
    Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
    """
    print(f"\nFinal Data::: ==>{final_data}")
    return final_data

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def get_rag_chain(llm: ChatOpenAI, knowledge_graph: Neo4jGraph):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    entity_chain = prompt | llm.with_structured_output(Entities)

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x: x["question"]),
    )
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | RunnableLambda(
                    lambda question: retriever(
                        question=question,
                        entity_chain=entity_chain,
                        knowledge_graph=knowledge_graph
                    )
                ),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    llm = load_llm()
    knowledge_graph = load_knowledge_graph()
    # generate_knowledge_graph("Vampires", llm, knowledge_graph)

    knowledge_graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    chain = get_rag_chain(llm, knowledge_graph)
    answer = chain.invoke({"question": "who is the most popular vampire?"})
    print("Answer: ", answer)

if __name__ == "__main__":
    main()
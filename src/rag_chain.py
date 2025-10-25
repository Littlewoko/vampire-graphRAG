from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from entities import Entities
from retrievers import retriever

def _format_chat_history(chat_history):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def get_rag_chain(llm, knowledge_graph):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ]
    )
    entity_chain = prompt | llm.with_structured_output(Entities)

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llm.__class__(temperature=0)
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    answer_prompt = ChatPromptTemplate.from_template(template)

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
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    return chain
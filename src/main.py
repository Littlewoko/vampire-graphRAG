from loaders import load_llm, load_knowledge_graph
from rag_chain import get_rag_chain
# from graph import generate_knowledge_graph # if needed

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
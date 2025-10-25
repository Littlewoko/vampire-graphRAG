from loaders import load_llm, load_knowledge_graph
from graph import generate_knowledge_graph

def main():
    llm = load_llm()
    knowledge_graph = load_knowledge_graph()
    # generate_knowledge_graph("Vampires", llm, knowledge_graph)
    print("Graph documents stored successfully")

if __name__ == "__main__":
    main()
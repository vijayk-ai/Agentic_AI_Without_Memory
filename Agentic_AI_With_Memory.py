import os
import faiss

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_tavily import TavilySearch

# -----------------------------
# API Keys
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    "Summarize these results:\n{docs}"
)

chain = prompt | llm

# -----------------------------
# Search Tool
# -----------------------------
search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=TAVILY_API_KEY
)

# -----------------------------
# Vector Store
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

dim = len(embeddings.embed_query("hello"))
index = faiss.IndexFlatL2(dim)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# -----------------------------
# Agent Function
# -----------------------------
def run_agent(query: str):
    print(f"\nRunning agent on query: {query}")

    # 1️⃣ Run search
    search_results = search_tool.invoke({"query": query})

    results = search_results.get("results", [])

    # 2️⃣ Extract real article content
    search_texts = [
        r.get("content", "")
        for r in results
        if r.get("content")
    ]

    if not search_texts:
        print("No valid content returned from search.")
        return

    # 3️⃣ Store in memory
    vectorstore.add_texts(search_texts)

    # 4️⃣ Summarize
    response = chain.invoke({
        "docs": "\n\n".join(search_texts)
    })

    print("Agent Response:\n", response.content)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    queries = [
        "Breakthrough AI technologies in 2026",
        "AI advancements in healthcare 2025-2026"
    ]

    for q in queries:
        run_agent(q)

"""
PageIndex RAG Agent (vectorless, tree-based document Q&A) - CLI for CFMS queries.

.env keys needed: PAGEINDEX_API_KEY and NVIDIA_API_KEY.
Optional tracing (LangSmith): LANGSMITH_TRACING=true, LANGSMITH_API_KEY=..., LANGSMITH_PROJECT=pageindex-rag.
"""

from rag_core import RagApp

def main():
    try:
        app = RagApp()
        print("RAG Agent Initialized!")
    except Exception as e:
        raise SystemExit(f"⚠️ Startup failed: {e}")
    with app:
        while True:
            try:
                question = input("Ask your question about the document (enter 'q' to quit):\n").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nThanks for using this agent!")
                break
            if question.lower() == "q":
                print("Thanks for using this agent!")
                break
            if not question:
                continue
            try:
                app.ask(question)
            except Exception as e:
                print(f"⚠️ Request failed: {e}\nTry again.\n")

if __name__ == "__main__":
    main()
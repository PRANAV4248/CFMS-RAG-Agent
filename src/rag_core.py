"""
rag_core.py – Vectorless RAG core.

Builds a PageIndex-backed LangGraph agent that answers questions about a PDF
using a tree-structured document index (no embeddings, no vector store).

Required .env keys: PAGEINDEX_API_KEY, NVIDIA_API_KEY
"""

import json
import os
import threading
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

_SRC_DIR  = os.path.dirname(__file__)
DOCS_DIR  = os.path.join(_SRC_DIR, "documents")
PDF_PATH  = os.path.join(DOCS_DIR, "CFMS_guide.pdf")
_CACHE    = os.path.join(DOCS_DIR, "pageindex_docs.json")
_TREE_DIR = os.path.join(DOCS_DIR, "trees")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
PRIMARY_MODEL   = "nvidia/nemotron-3-super-120b-a12b"

# Prevents concurrent sessions from uploading the same PDF simultaneously.
_DOC_LOCK = threading.Lock()


class Answer(BaseModel):
    is_doc_query: bool = Field(
        default=True,
        description="True when answering a document question; False for greetings / off-topic.",
    )
    answer: str = Field(
        description="Final answer with page numbers, or a conversational reply for greetings.",
    )
    node_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the sections actually read to produce this answer.",
    )


class RagApp:
    """Vectorless RAG agent backed by PageIndex."""

    def __init__(self, pdf_path: str = PDF_PATH) -> None:
        from pageindex import PageIndexClient
        from pageindex.utils import structure_to_list
        from langchain.rate_limiters import InMemoryRateLimiter

        missing = [k for k in ("PAGEINDEX_API_KEY", "NVIDIA_API_KEY") if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

        self.pdf_path = pdf_path
        self.client   = PageIndexClient(index="cloud")

        with _DOC_LOCK:
            self.doc_id = self._get_or_create_doc_id()
            self.tree   = self._get_or_create_tree()

        self.nodes = {n["node_id"]: n for n in structure_to_list(self.tree)}

        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.5,
            check_every_n_seconds=0.1,
            max_bucket_size=5,
        )
        self.rate_limiter.available_tokens = self.rate_limiter.max_bucket_size

        self.agent = self._build_agent()
        self.reset()

    def reset(self) -> None:
        """Start a fresh conversation (new memory thread)."""
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def close(self) -> None:
        getattr(self.client, "close", lambda: None)()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _get_or_create_doc_id(self) -> str:
        key = os.path.basename(self.pdf_path)
        try:
            with open(_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except FileNotFoundError:
            cache = {}

        if key in cache:
            return cache[key]

        doc_id = self.client.submit_document(self.pdf_path, wait=True)["doc_id"]
        cache[key] = doc_id
        with open(_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        return doc_id

    def _get_or_create_tree(self) -> list:
        os.makedirs(_TREE_DIR, exist_ok=True)
        tree_file = os.path.join(_TREE_DIR, f"tree_{self.doc_id}.json")

        if os.path.exists(tree_file):
            with open(tree_file, "r", encoding="utf-8") as f:
                return json.load(f)

        tree = self.client.get_tree(self.doc_id, node_summary=True).get("result", [])
        with open(tree_file, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2)
        return tree

    def _build_agent(self):
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            ClearToolUsesEdit,
            ContextEditingMiddleware,
            ModelRetryMiddleware,
        )
        from langchain.agents.structured_output import ToolStrategy
        from langchain.chat_models import init_chat_model
        from langgraph.checkpoint.memory import InMemorySaver

        model = init_chat_model(
            model=PRIMARY_MODEL,
            model_provider="openai",
            base_url=NVIDIA_BASE_URL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0,
            rate_limiter=self.rate_limiter,
        )

        outline = "\n".join(
            f"- [{n['node_id']}] {n['title']} (pages {n['start_index']}-{n['end_index']})"
            for n in self.nodes.values()
        )

        system = f"""# BASE TOOL INSTRUCTIONS
{self.client.agent_instructions()}

# DOCUMENT CONTEXT
{self.client.document_context(self.doc_id)}

# CFMS OVERRIDE — these instructions take precedence over everything above
You are an expert assistant for the Cryogen-Free Measurement System (CFMS) user guide.
This document is a 34-page internship report from IIT (BHU) Varanasi by Pranav Choubey, covering:
cryogen-free measurement, cryo-cooler system, cryostat, superconducting magnet, Variable Temperature Insert (VTI), electronics rack, LabVIEW measurement software, operating procedures, safety and warm-up guidelines, maintenance, troubleshooting, applications, and system specifications.

Document Table of Contents (use this to locate sections — do NOT call get_document_structure()):
{outline}

RULES:
- The document is fully indexed above. Use the Table of Contents to choose which pages to read with get_page_content(). Never call get_document_structure() or browse_documents() or search_documents().
- Always cite the specific section name and page numbers in your answer.
- In node_ids, list all section IDs (e.g. ['0023', '0024']) from the Table of Contents that were referenced or read to produce your answer.
- For greetings or questions unrelated to CFMS: set is_doc_query=false and leave node_ids empty."""

        return create_agent(
            model=model,
            tools=self.client.agent_tools(),
            system_prompt=system,
            response_format=ToolStrategy(Answer),
            checkpointer=InMemorySaver(),
            middleware=[
                ModelRetryMiddleware(max_retries=2),
                ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=6000, keep=2)]),
            ],
        )

    def ask(self, question: str, verbose: bool = True) -> "Answer | None":
        """Send a question to the agent and return a structured Answer (or None on hard failure)."""
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=self.config,
        )
        result = response.get("structured_response")

        if verbose:
            self._print_result(result)

        return result

    def _print_result(self, result) -> None:
        if result is None:
            print("\n⚠️  No structured response produced.\n")
            return
        if not result.is_doc_query:
            print(f"\n💬 {result.answer}\n")
            return
        print(f"\n📝 Answer: {result.answer}\n")
        if result.node_ids:
            print("📚 Sections used:")
            for nid in result.node_ids:
                node = self.nodes.get(nid)
                if node:
                    print(f"   [{nid}] {node['title']} (pages {node['start_index']}-{node['end_index']})")
                else:
                    print(f"   [{nid}] (not found in index)")
        else:
            print("⚠️  Doc question answered without reading any section – treat as unverified.\n")
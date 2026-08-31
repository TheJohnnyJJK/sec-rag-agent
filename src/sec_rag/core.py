"""SecRag — one place where all pipeline setup happens.

Everything the RAG flow needs (vector store, BM25 index, reranker, Anthropic
client, rules prompt) is built exactly once in ``__init__``. The notebook,
the MCP server, and the agent all construct one ``SecRag`` and call its
methods — no setup is duplicated anywhere else.

The class methods CALL the existing pipeline functions
(``answer_question``, ``retrieve``, ``rerank`` …) — they do not reimplement them.
"""

import ast
import operator
import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
from dotenv import find_dotenv, load_dotenv
from edgar import Company, set_identity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from sec_rag.embed import build_bm25_index
from sec_rag.generate import answer_question, rules, agent_rules
from sec_rag.ingest import process_filing
from sec_rag.raptor import build_raptor_summaries

# Repo-root-relative so it resolves to the same directory server.py and the
# notebook use, regardless of where the process is launched from.
#   parents[0] = sec_rag/  ·  parents[1] = src/  ·  parents[2] = repo root
VECTOR_STORE_PATH = str(Path(__file__).resolve().parents[2] / "vectorStore")

# Tool schemas the agent's Claude loop reads (lifted verbatim from the notebook).
CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Evaluate a math expression. Use for any arithmetic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '34550 / 416161'"}
        },
        "required": ["expression"],
    },
}

SEARCH_TOOL = {
    "name": "search_filings",
    "description": "Look up a fact in Apple's 2025 10-K filing. Use for any question about Apple's financials or business — revenue, R&D, net income, etc.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "e.g. 'What was Apple's total revenue in 2025?'"}
        },
        "required": ["question"],
    },
}

# The calculator tool runs on model-generated strings. Evaluate them through a
# whitelisted AST walk (numbers + the five arithmetic ops + parentheses), never
# the builtin eval — that would be arbitrary code execution on LLM output.
_ARITH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_arithmetic(expression: str):
    """Evaluate a plain arithmetic expression. Raises ValueError on anything else."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {expression!r}")

    return _eval(ast.parse(expression, mode="eval").body)


class SecRag:
    """Holds the fully wired-up pipeline. Build once, reuse for every query."""

    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        # --- API keys from .env ---
        load_dotenv(find_dotenv(usecwd=True))

        # --- Persisted Chroma vector store (load from disk, no re-embedding) ---
        chroma_client = chromadb.PersistentClient(path=vector_store_path)
        self.vectorestore = chroma_client.get_or_create_collection(name="vectorStore")

        # --- Re-ingest the Apple 10-K to rebuild fixed_chunks ---
        # BM25 has no on-disk persistence, so the chunk pool must be rebuilt in
        # process to feed it. Same chunking params as the original ingestion.
        # EDGAR wants a real "Name email" contact string on every request; set
        # SEC_IDENTITY in .env. The fallback keeps things working for a quick try.
        set_identity(os.getenv("SEC_IDENTITY", "sec-rag example@example.com"))
        filing = Company("AAPL").get_filings(form="10-K")[0]
        tenk = filing.obj()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.fixed_chunks = process_filing(tenk, splitter)

        # --- BM25 keyword index over that chunk pool ---
        self.bm25 = build_bm25_index(self.fixed_chunks)

        # --- Cross-encoder reranker ---
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # --- Anthropic client ---
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # --- Grounding rules prompt ---
        self.rules = rules

        # --- Agent rules prompt ---
        self.agent_rules = agent_rules
        
        # --- Embedder ---
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, question: str, rules=None, rerank_top_n: int = 10) -> str:
        """Full RAG flow for one question → grounded answer string.

        Thin wrapper over the existing ``answer_question`` pipeline.
        Pass `rules` to override the default verbose prompt (the agent passes
        the terse `agent_rules` so its internal lookups stay short).
        `rerank_top_n` is the reranker cutoff (default 10, so RAPTOR summaries
        survive into the context).
        """
        return answer_question(
            question,
            self.vectorestore,
            self.bm25,
            self.fixed_chunks,
            self.reranker,
            self.client,
            rules or self.rules,          # use passed rules, else the verbose default
            rerank_top_n,
        )

    def build_raptor_summaries(
        self,
        *,
        pca_dims: int = 10,
        n_range: range = range(2, 15),
        threshold: float = 0.10,
        force: bool = False,
    ):
        """Cluster the chunk pool, summarize each cluster with Haiku, and add the
        summaries to the vector store as extra retrievable documents.

        Thin wrapper over ``sec_rag.raptor.build_raptor_summaries`` — same
        pattern as ``search`` over ``answer_question``. Idempotent: existing
        ``type: "summary"`` rows are left in place unless ``force=True``.
        """
        return build_raptor_summaries(
            self.fixed_chunks,
            self.embedder,
            self.vectorestore,
            self.client,
            pca_dims=pca_dims,
            n_range=n_range,
            threshold=threshold,
            force=force,
        )

    def run_agent(self, question: str, max_steps: int = 8):
        """Agentic tool-use loop over two tools: calculator + search_filings.

        Returns (answer, steps, usage). steps is a list of
        {"tool", "input", "result"} dicts; usage totals tokens across all calls.
        Capped at max_steps.
        """
        tools = [CALCULATOR_TOOL, SEARCH_TOOL]
        messages = [{"role": "user", "content": question}]
        steps: list[dict] = []
        total_input = 0
        total_output = 0

        for _ in range(max_steps):
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                tools=tools,
                messages=messages,
            )

            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            if response.stop_reason == "end_turn":
                answer = "".join(b.text for b in response.content if b.type == "text")
                usage = {"input_tokens": total_input, "output_tokens": total_output}
                return answer, steps, usage

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == "calculator":
                        try:
                            result = str(safe_arithmetic(block.input["expression"]))
                        except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
                            result = f"calculator error: {e}"
                    elif block.name == "search_filings":
                        result = self.search(block.input["question"], rules=self.agent_rules)  # terse
                    else:
                        result = f"Unknown tool: {block.name}"

                    steps.append({"tool": block.name, "input": block.input, "result": result})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

        # Ran max_steps without finishing — return three values to match the normal return.
        usage = {"input_tokens": total_input, "output_tokens": total_output}
        return "Agent stopped: hit max steps.", steps, usage
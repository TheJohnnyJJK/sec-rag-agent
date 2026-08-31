from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import Context, FastMCP

from sec_rag.core import SecRag


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    # All setup lives in SecRag.__init__ — one place, no duplication.
    rag = SecRag()
    yield {"rag": rag}


mcp = FastMCP("sec-rag", lifespan=app_lifespan)


@mcp.tool()
def search_filings(question: str, ctx: Context) -> str:
    """Answer questions about Apple's 2025 annual report (SEC Form 10-K). Use this for factual questions about Apple's business and financials from that filing — revenue, net income, R&D spending, employee count, products, business segments, risk factors, and similar. Answers are grounded in the filing text with citations, and it will say when the filing doesn't contain the answer."""
    rag: SecRag = ctx.lifespan_context["rag"]
    return rag.search(question)


if __name__ == "__main__":
    mcp.run()

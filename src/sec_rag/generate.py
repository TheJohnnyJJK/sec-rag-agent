from sec_rag.retrieve import retrieve, rerank

rules = """You are a financial analyst assistant answering questions about SEC filings. You may ONLY use information that physically appears in the <excerpts> provided in the user message. You have no other knowledge.

ABSOLUTE RULES — these override everything else:

1. GROUNDING: Every fact, number, date, name, and statistic in your answer MUST appear verbatim in the provided <excerpts>. If it is not literally written in the excerpts, you do not know it. Treat yourself as having zero outside knowledge about this company or any financial data.

2. NUMBERS: Never state any figure, dollar amount, percentage, or statistic unless that exact number appears in the excerpts. You may know financial figures from elsewhere — you must ignore that knowledge completely. A number you "remember" but cannot see in the excerpts does not exist.

3. CITATIONS: Only cite a source that is actually shown in the excerpts (the [Company | Form | Period] label above each excerpt). NEVER invent, assume, or reference any source, statement, table, or section that is not present in the excerpts. Do not cite "Consolidated Statements of Operations" or any financial statement unless that text is in the excerpts.

4. REFUSAL: If the excerpts do not contain the specific information needed to answer, respond with exactly: "I don't have enough information" — and nothing else. This applies even if you are certain you know the answer from other sources. Refusing is always correct when the excerpts lack the answer. It is far better to refuse than to guess.

5. VERIFY BEFORE ANSWERING: Before you write any fact or number, silently confirm it appears in the excerpts. If you cannot point to where it appears, do not include it. If the core of the question cannot be answered from the excerpts, refuse per rule 4.

6. NO OUTSIDE REASONING: Do not supplement, infer, or fill gaps using general knowledge about the company, its products, or its finances. Report only what the excerpts state.

You report what the filings say. You do not analyze, estimate, or add information. When in doubt, refuse."""

agent_rules = """You answer questions about SEC filings using ONLY the information in the <excerpts> provided. You have no outside knowledge.

Reply with just the figure and a short label — e.g. "R&D expense 2025: $34,550 million." No preamble, no "according to the excerpts," no explanation, no citations.

If the specific number or fact is not in the excerpts, reply with exactly: "Not found."

Never state any figure that does not appear verbatim in the excerpts."""


def build_context(chunks):
    """Format the chosen chunks into one labelled block the model can cite from."""
    blocks = [
        c["meta"]["Company"] + " | " + c["meta"]["Form"] + " | " + c["meta"]["Period"] + "\n" + c["text"]
        for c in chunks
    ]
    return "\n\n".join(blocks)


def generate(question, context, client, rules):
    """Send the grounded prompt to Claude and return the answer text."""
    prompt = f"""<excerpts>
{context}
</excerpts>

Question: {question}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=rules,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def answer_question(question, vectorestore, bm25, fixed_chunks, reranker, client, rules, rerank_top_n=10):
    """Orchestrate the full RAG flow: retrieve → rerank → build context → generate.

    ``rerank_top_n`` is the reranker cutoff — how many chunks survive into the
    context. Kept at 10 (not 5) so RAPTOR cluster summaries, which rerank lower
    than pinpoint leaf chunks, still reach the answer.
    """
    candidates = retrieve(question, vectorestore, bm25, fixed_chunks)
    top = rerank(question, candidates, reranker, rerank_top_n)
    context = build_context(top)
    return generate(question, context, client, rules)

# sec-rag

A retrieval-augmented question-answering pipeline over a single SEC 10-K filing
(Apple's fiscal 2025 annual report), built from parts rather than a framework so
each stage can be measured on its own. It does hybrid retrieval, cross-encoder
reranking, grounded generation with citations, an LLM-as-judge eval harness, an
MCP server, a small tool-using agent, and a RAPTOR-style summary layer on top of
the chunk index.

The filing is chunked into 271 passages. Everything downstream operates on that
pool.

## How it works

**Ingest.** `edgartools` pulls the latest Apple 10-K and parses it into sections
(Item 1, Item 1A, and so on). Each section is split on its own paragraph breaks
first, and only oversized blocks (dense tables, long paragraphs) fall through to
a recursive character splitter. Every chunk carries its company, form, period,
and source item as metadata.

**Index.** Chunks are embedded with `all-MiniLM-L6-v2` and written to a
persistent Chroma collection. A BM25 keyword index is built over the same pool in
memory. The two are kept deliberately separate — they have opposite strengths.

**Retrieve.** A query runs against both indexes. Vector search matches on
meaning; BM25 matches on exact tokens and catches the literal figures ("166,000",
"$34,550") that a dense embedding tends to smooth over. The two result sets are
merged and de-duplicated on chunk text.

**Rerank.** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores every merged
candidate against the query and keeps the top `rerank_top_n` (default 10). This
is the cutoff that decides what the model actually sees.

**Generate.** The surviving chunks are formatted into a labelled context block
and sent to Claude Haiku with a system prompt that forbids any fact not present
in the excerpts. If the answer isn't in the context, the model is told to reply
"I don't have enough information" rather than reach for outside knowledge.

**RAPTOR layer.** The leaf chunks are clustered and each cluster is summarised,
so a broad question can retrieve one topic-level passage instead of hoping the
right handful of leaves surface. The chunk embeddings are reduced to 10
dimensions with PCA, a Gaussian mixture model is fit for each cluster count from
2 to 14, and the count with the lowest BIC wins (3, on this filing). Membership
is soft: a chunk joins every cluster it scores at least 0.10 in, so a passage
that straddles two topics lands in both. Each bucket is summarised by Haiku, and
the summaries are embedded and added back into the same Chroma collection with
ids `summary_0..N` and `type: "summary"` metadata. Re-running is a no-op unless
called with `force=True`.

**Agent.** `SecRag.run_agent` is a short tool-use loop with two tools —
`search_filings` (the RAG pipeline above, with a terser prompt) and a
`calculator`. It handles questions that need more than one lookup plus
arithmetic, like R&D as a percentage of revenue.

**MCP server.** `server.py` exposes `search_filings` as an MCP tool via
`fastmcp`, so the pipeline can be called from an MCP client. All setup happens
once in `SecRag.__init__`; the server, the notebook, and the agent all share
that one object.

## Layout

```
src/sec_rag/
  ingest.py     chunker, process_filing
  embed.py      shared tokenizer, embedding, BM25 index
  retrieve.py   hybrid retrieve, rerank
  generate.py   grounding prompt, build_context, generate, answer_question
  raptor.py     summarize_cluster, build_raptor_summaries
  evaluate.py   judge prompts, eval_set, run_eval, agent_eval, run_agent_eval
  core.py       SecRag — builds everything once, wraps the pipeline functions
server.py       MCP server
notebooks/
  explore.ipynb driver: builds a SecRag, runs search, eval, agent, RAPTOR
FINDINGS.md     root-cause log for retrieval and eval bugs
```

The modules hold the logic. `core.py` wires it together and calls those
functions — it doesn't reimplement them. The notebook is a consumer, not the
container.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
cp .env.example .env      # then add ANTHROPIC_API_KEY and SEC_IDENTITY
```

`uv sync` creates `.venv` from the lockfile and installs the package in editable
mode. `requirements.txt` is kept in step with `pyproject.toml` for anyone using
plain pip instead.

The Chroma store under `vectorStore/` is a build artifact and is not committed —
it is regenerated from the filing. Build it once from the bootstrap cell in
`notebooks/explore.ipynb` (the one marked "one-time bootstrap"), which embeds
every chunk and writes the collection to disk. `SecRag()` loads that store on
startup and does not re-embed.

## Running it

```python
from sec_rag.core import SecRag

rag = SecRag()
rag.search("what does apple even make")
rag.run_agent("What was Apple's R&D spending as a percentage of revenue in 2025?")
rag.build_raptor_summaries()          # force=True to rebuild
```

MCP server:

```
uv run python server.py
```

Eval:

```python
from sec_rag.evaluate import eval_set, run_eval, judge_rules
from sec_rag.evaluate import agent_eval, run_agent_eval, judge_rules_agent

run_eval(eval_set, rag.vectorestore, rag.bm25, rag.fixed_chunks,
         rag.reranker, rag.client, rag.rules, judge_rules)
run_agent_eval(agent_eval, rag, judge_rules_agent)
```

## Eval

`eval_set` is 20 questions: 10 phrased the way a person actually types ("apple
revenue", "how much on r&d", "how many people work at apple") and 10 phrased
formally, including two that the filing genuinely can't answer and the system is
expected to refuse (the CEO's exact salary, 2019 net sales). A Haiku judge scores
each answer against a reference on meaning, not wording, and checks that refusals
line up on both sides.

The full pipeline currently scores 20/20. The agent eval — five questions that
each need two or more lookups and a calculation — scores 5/5.

## What the measurements showed

**Each retrieval stage earned its place.** Measured one change at a time on the
eval set as it stood at the time (24 questions then; later consolidated to the
20 above):

| Stage | Score |
|---|---|
| Vector search only | 17 / 24 |
| + reranking | 20 / 24 |
| + hybrid BM25 search | 23 / 24 |
| + consistent BM25 tokenization | 24 / 24 |

The last row was a bug, not a feature. BM25 matches tokens by exact string
equality, and the corpus was tokenized one way (`.split()`, case kept,
punctuation glued on) while the query was tokenized another (`\w+`, lowercased).
"Research" never matched "research"; "development$34,550" never matched
"development". The failing chunk sat at rank 27 instead of rank 0. One shared
`tokenize()` function fixed it. Full write-up in `FINDINGS.md`.

**RAPTOR summaries and the reranker pull against each other.** A cluster summary
is broad by construction, so the cross-encoder scores it below a leaf chunk that
contains the exact phrasing of the question. At a top-5 cutoff the summaries were
getting squeezed out before they reached the model. Raising `rerank_top_n` to 10
leaves room for them to survive alongside the pinpoint chunks. It is a real
trade-off — a wider context window in exchange for keeping the topic-level
passage in play — not a free win.

**The agent takes the long way sometimes.** All five multi-step questions pass,
but the tool path isn't always minimal: the agent will occasionally make an
extra `search_filings` call for a figure it already retrieved before moving on to
the `calculator`. Runs land between three and five steps and roughly 3,000 to
4,500 tokens per question. Good enough for the eval; worth watching if step count
starts to matter.

## Notes

The agent's `calculator` tool evaluates expressions produced by the model. It
does this through a whitelisted AST walk in `core.py` — numbers, the five
arithmetic operators, and parentheses, nothing else — not the builtin `eval`.

EDGAR asks every caller to identify itself with a name and email. Set
`SEC_IDENTITY` in `.env` ("Your Name you@example.com"). If it's unset the code
falls back to a placeholder, which EDGAR may rate-limit.

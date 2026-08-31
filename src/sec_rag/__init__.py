"""sec_rag — a hybrid RAG pipeline over SEC filings, built from scratch.

Modules:
    ingest    — load a filing, chunk it, attach metadata
    embed     — embedding model + vector store + BM25 keyword index
    retrieve  — hybrid retrieval (vector + BM25 merge) and reranking
    generate  — grounded prompt, context building, answer generation
    raptor    — cluster the chunk pool, summarize each cluster, index the summaries
    evaluate  — LLM-as-judge scoring: run_eval (RAG) and run_agent_eval (agent)
"""

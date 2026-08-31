from sec_rag.embed import tokenize


def retrieve(question, vectorestore, bm25, fixed_chunks, n_results=20):
    # HYBRID RETRIEVAL: run two searches with opposite strengths and merge them.
    #   - vector search  → matches by MEANING (good for concepts, misses buried literal facts)
    #   - BM25 keyword   → matches by EXACT WORDS (catches "166,000 employees" the vector blurs over)
    # Merging both means each covers the other's blind spot; the reranker sorts the pool afterward.

    # --- 1. VECTOR SEARCH (semantic) ---
    # Chroma returns parallel lists (documents + metadatas), nested one level per query ([0]).
    # zip them back into one {"text", "meta"} dict per chunk so both searches share the same shape.
    results = vectorestore.query(query_texts=[question], n_results=n_results)
    vector_chunks = [
        {"text": doc, "meta": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    # --- 2. BM25 SEARCH (keyword) ---
    # get_scores → one keyword-match score per chunk, aligned by position with fixed_chunks.
    bm25_scored = bm25.get_scores(tokenize(question))
    # argsort gives the positions that sort the scores low→high; [::-1] flips to best-first.
    ranked = bm25_scored.argsort()[::-1]
    # Take the top 20 positions and pull those chunks, reshaping to the same {"text", "meta"} form.
    bm25_chunks = [
        {"text": fixed_chunks[i]["text"],
         "meta": {"Company": fixed_chunks[i]["Company"],
                  "Form": fixed_chunks[i]["Form"],
                  "Period": fixed_chunks[i]["Period of report"]}}
        for i in ranked[:20]
    ]

    # --- 3. MERGE + DEDUPE ---
    # A chunk can appear in BOTH searches. Keying a dict on the chunk text collapses duplicates:
    # identical text overwrites the same key, so each unique chunk survives exactly once.
    merged = {}
    for c in vector_chunks + bm25_chunks:
        merged[c["text"]] = c
    return list(merged.values())   # combined, deduped candidate pool → goes to rerank


def rerank(question, candidates, reranker, top_n=10):
    """Score each candidate against the question with the cross-encoder, keep the best top_n."""
    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, score in zip(candidates, scores):
        c["score"] = score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]

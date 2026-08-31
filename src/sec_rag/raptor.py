"""RAPTOR-style hierarchical indexing: cluster the chunk pool, summarize each
cluster with Haiku, and add those summaries back into the vector store as extra
retrievable "documents".

Why: a leaf chunk is ~500 chars and only ever answers a local question. A
cluster summary sits one level up — it captures a whole topic (all the revenue
chunks, all the supply-chain-risk chunks) in one passage, so a broad or vague
query ("what does apple even make") can retrieve the summary instead of hoping
the right five leaves surface.

Flow (lifted from the notebook, parameters exposed as arguments):
    embed every chunk -> PCA to `pca_dims` -> GMM, cluster count auto-picked by
    BIC over `n_range` -> predict_proba -> soft-assign each chunk to every
    cluster it scores >= `threshold` in (a chunk can land in several) ->
    summarize each bucket with Haiku -> embed the summaries -> add to Chroma
    with stable ids summary_0/1/2... and metadata {"type": "summary", ...}
    matching the existing chunk metadata shape.

Uses scikit-learn (PCA, GaussianMixture). NOT umap-learn — it crashes the
kernel on this data.

``SecRag.build_raptor_summaries`` is a thin wrapper over
``build_raptor_summaries`` here, the same way ``SecRag.search`` wraps
``answer_question``.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from sec_rag.embed import embedding


def summarize_cluster(chunks, client):
    """Collapse one cluster's chunks into a single grounded overview via Haiku."""
    # join all the chunk texts in this cluster into one big blob
    combined = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""Summarize the following excerpts from Apple's 2025 10-K filing into a concise overview that captures the main topics, facts, and figures. Keep it factual and grounded in the text.

{combined}

Summary:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_raptor_summaries(
    fixed_chunks,
    embedder,
    vectorestore,
    client,
    *,
    pca_dims: int = 10,
    n_range: range = range(2, 15),
    threshold: float = 0.10,
    force: bool = False,
):
    """Build cluster summaries and add them to the vector store.

    Idempotent: if the store already holds ``type: "summary"`` items this
    returns them untouched (pass ``force=True`` to delete and rebuild). New
    summaries are written with stable ids ``summary_0``, ``summary_1``, … so a
    rebuild overwrites the same slots rather than piling up duplicates.

    Returns a list of ``{"id", "size", "summary"}`` dicts, one per cluster.
    (``size`` is ``None`` for summaries returned from an existing store — the
    bucket membership that produced them is not persisted.)
    """
    # --- Idempotency guard: don't re-summarize (3+ Haiku calls) if already done ---
    existing = vectorestore.get(where={"type": "summary"}, include=["documents"])
    if existing["ids"]:
        if not force:
            print(
                f"[raptor] {len(existing['ids'])} summaries already in store "
                f"({', '.join(existing['ids'])}); skipping. Pass force=True to rebuild."
            )
            return [
                {"id": sid, "size": None, "summary": doc}
                for sid, doc in zip(existing["ids"], existing["documents"])
            ]
        # force: clear every existing summary (best_n may have changed) before rebuild
        vectorestore.delete(ids=existing["ids"])

    # --- 1. Embed every chunk (locally; we don't mutate the shared fixed_chunks dicts) ---
    # Same model the store was built with, so PCA/GMM see the same geometry retrieval does.
    embeddings = np.array(embedding([c["text"] for c in fixed_chunks], embedder))

    # --- 2. PCA down to pca_dims ---
    # GMM in 384-D is noisy and slow; 10 dense dims keep the cluster structure that matters.
    reduced = PCA(n_components=pca_dims).fit_transform(embeddings)

    # --- 3. Auto-pick the cluster count by BIC over n_range ---
    # BIC rewards fit, penalizes free parameters -> lowest BIC = best cluster count
    # without us hand-guessing it.
    bics = []
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(reduced)
        bics.append(gmm.bic(reduced))
    best_n = list(n_range)[int(np.argmin(bics))]
    print(f"[raptor] best number of clusters (BIC over {n_range.start}..{n_range.stop - 1}): {best_n}")

    # --- 4. Refit at best_n, get soft memberships ---
    gmm = GaussianMixture(n_components=best_n, random_state=42)
    gmm.fit(reduced)
    probs = gmm.predict_proba(reduced)  # row i aligns with fixed_chunks[i]

    # --- 5. Threshold into buckets (a chunk joins every cluster it scores >= threshold in) ---
    clusters = {i: [] for i in range(best_n)}
    for chunk_idx, row in enumerate(probs):
        for cluster_idx, prob in enumerate(row):
            if prob >= threshold:
                clusters[cluster_idx].append(fixed_chunks[chunk_idx])
    for i in range(best_n):
        print(f"[raptor] cluster {i}: {len(clusters[i])} chunks")

    # --- 6. Summarize each bucket with Haiku ---
    summaries = [summarize_cluster(clusters[i], client) for i in range(best_n)]

    # --- 7. Embed the summaries and add them to Chroma ---
    # Metadata mirrors the existing chunk rows ({"Company","Period","Form"}) plus a
    # "type": "summary" tag, so build_context can cite a retrieved summary unchanged
    # and the idempotency guard above can find them next time.
    meta_src = fixed_chunks[0]
    base_meta = {
        "Company": meta_src["Company"],
        "Period": meta_src["Period of report"],
        "Form": meta_src["Form"],
    }
    summary_embeddings = np.array(embedding(summaries, embedder))
    ids = [f"summary_{i}" for i in range(best_n)]
    vectorestore.add(
        ids=ids,
        embeddings=summary_embeddings.tolist(),
        documents=summaries,
        metadatas=[{**base_meta, "type": "summary"} for _ in range(best_n)],
    )
    print(f"[raptor] added {best_n} summaries to the store: {', '.join(ids)}")

    return [
        {"id": ids[i], "size": len(clusters[i]), "summary": summaries[i]}
        for i in range(best_n)
    ]

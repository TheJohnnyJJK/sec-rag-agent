import re

from rank_bm25 import BM25Okapi


# Shared tokenizer for BM25 — MUST be used on both the corpus (build_bm25_index) and the
# query (retrieve) side. BM25 matches tokens by exact string equality, so any drift between
# the two (case, punctuation) silently drops matches instead of erroring.
def tokenize(text):
    return re.findall(r"\w+", text.lower())


# Turn a piece of text (or list of texts) into embedding vector(s) using the given model.
# embedder is passed in (dependency injection) so the model can be swapped without touching this.
def embedding(list_of_chunks, embedder):
    embedding = embedder.encode(list_of_chunks)   # SentenceTransformer.encode → vector(s)
    return embedding


def build_bm25_index(fixed_chunks):
    """Build a BM25 keyword index over the chunk pool, aligned by position with fixed_chunks."""
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in fixed_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Findings log

Root-cause record for bugs and retrieval/eval failures in this pipeline. Check
here first when a symptom looks familiar — tokenization and hybrid-retrieval
bugs especially tend to recur in slightly different shapes.

When you close out an investigation (especially one that changed the eval
score), add an entry. Keep entries short: symptom → root cause → fix → what
would have caught it sooner.

---

## 2026-08-26 — `retrieve()` NameError: `re` not defined

**Symptom:** Calling `sec_rag.retrieve.retrieve()` raised
`NameError: name 're' is not defined` from a fresh kernel.

**Root cause:** `src/sec_rag/retrieve.py` called `re.findall(...)` inside
`retrieve()` but never imported `re` in that module. It had previously "worked"
in a long-lived notebook kernel only because some other import path had left
`re` bound in that module's namespace at the time — not because the code was
correct. A restarted kernel exposed it immediately.

**Fix:** Added `import re` (later replaced by `from sec_rag.embed import
tokenize`, see next entry) to `retrieve.py`.

**Lesson:** A function relying on a name it doesn't import can silently work
for an entire session and then break on the next kernel restart. When
debugging "used to work," restart the kernel and re-run top-to-bottom before
trusting any live state.

---

## 2026-08-26 — Eval failure on R&D question: BM25/query tokenization mismatch

**Symptom:** Eval scored 23/24. The one failure — "How much did Apple spend
on research and development in 2025?" — got a refusal ("I don't have enough
information") even though the correct chunk (`$34,550 million`) exists in
`fixed_chunks`.

**Investigation path (see [[retrieval-debug skill]] for the general method):**
1. Confirmed it wasn't a rerank/generation problem — the answer chunk never
   appeared in the merged candidate pool at all (retrieval-stage failure).
2. Checked full-corpus rank: the best answer chunk ranked 27th/271 in BM25
   and 37th/271 in vector search — both just past the hardcoded top-20
   cutoffs in `retrieve()`. Initially looked like a "raise n_results" fix.
3. Went one level deeper per user request and compared actual tokens: BM25
   matches by **exact token string equality**. The corpus was tokenized with
   `chunk["text"].split()` (case preserved, punctuation glued to adjacent
   words) while the query was tokenized with
   `re.findall(r"\w+", question.lower())` (lowercased, punctuation-stripped).
   Of the query's 8 substantive tokens, only 3 (`and`, `in`, `2025` — the
   last one by accidental match elsewhere in the chunk) survived to match the
   corpus. `"Research"` (capitalized) never matched `"research"`;
   `"development$34,550"` (glued to the dollar figure) never matched
   `"development"`.

**Root cause:** Two independent, differently-written tokenizers for the two
sides of one BM25 index (`build_bm25_index` in `embed.py` vs. the query-side
call in `retrieve.py`). BM25 has no fuzzy fallback — a token mismatch is a
silent zero, not an error, so this kind of bug produces a plausible-looking
but degraded score rather than a crash.

**Fix:** Extracted one shared `tokenize(text)` (lowercase + `re.findall(r"\w+",
...)`) into `embed.py`, used by both `build_bm25_index` and `retrieve`.
Rebuilding BM25 with consistent tokenization alone (no change to n_results or
the top-20 cutoff) moved the best answer chunk from rank 27 → rank 0.
Full eval: **24/24**, no regressions on the other 23 questions.

**Lesson:** Whenever a retrieval index and its query path tokenize
independently, check they use the *identical* function, not just "similar"
logic — case-folding and punctuation handling differences are invisible until
you diff the actual token lists for a specific failing example. Widening a
retrieval cutoff (`n_results`) can mask this class of bug — the failing chunk
`rank` may look "just barely" too deep and lure you into overprovisioning the
candidate pool instead of fixing the mismatch at its source. If a future eval
failure shows the right chunk near-but-outside the top-k cutoff, check
tokenization consistency before touching depth.

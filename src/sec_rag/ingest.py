# Split one section's text into retrieval-sized chunks.
# Strategy: use the document's own structure (\n\n block breaks) first — cheap and respects
# human-authored boundaries — and only escalate to the recursive splitter for rare oversized blocks.
def chunker(section, splitter):

    chunks = []      # finished chunks collected here
    current = ""     # "bucket" we accumulate small blocks into until it's big enough to flush

    # Clean non-breaking spaces (\xa0) → normal spaces so text is uniform before splitting
    cleaned_chunks = section.replace("\xa0", " ")

    # Walk the section one \n\n-delimited block at a time
    for chunk in cleaned_chunks.split("\n\n"):

        if len(chunk) > 2000:
            # ESCALATION: a single block is huge (usually a dense table / monster paragraph).
            # Hand it to RecursiveCharacterTextSplitter to break it down, and add the pieces.
            recursive_chunk = splitter.split_text(chunk)
            chunks.extend(recursive_chunk)   # extend (not append) so pieces go in individually
        else:
            # NORMAL PATH: glue this block onto the bucket. Small fragments (lone headers, etc.)
            # merge in automatically this way.
            current = current + " " + chunk
            # Once the bucket passes ~500 chars, flush it as one chunk and start a fresh bucket
            if len(current) > 500:
                chunks.append(current)
                current = ""

    # After the loop, whatever's left in the bucket is the final chunk — don't lose the tail
    if current:
        chunks.append(current)

    # Trim leading/trailing whitespace on every chunk in one pass
    chunks = [c.strip() for c in chunks]

    return chunks


# Take a parsed filing and turn it into a flat list of chunk dicts, each tagged with metadata.
# Metadata matters because once chunks are pooled in the vector store (many sections, later many
# filings), each chunk must carry where it came from — for filtering and for citations.
def process_filing(tenk, splitter):
    results = []

    # Filing-level metadata is the SAME for every chunk in this filing, so read it once here
    # (outside the loops) rather than re-fetching per chunk.
    company = tenk.company
    period = tenk.period_of_report
    form = tenk.form

    # Outer loop: go through each section (Item 1, Item 1A, ...). item_name is the section name.
    for item_name in tenk.items:
        # Chunk that one section's text into a list of chunk strings (splitter handles oversized blocks)
        chunks = chunker(tenk[item_name], splitter)

        # Inner loop: wrap each chunk string in its own dict with full metadata + the text itself.
        # This both flattens (no nested lists) and tags every chunk in one pass.
        for chunk in chunks:
            dictionary = {
                "complete": company + " " + form + " " + period + " " + " " + chunk,
                "Company": company,
                "Period of report": period,
                "Form": form,
                "item": item_name,   # per-chunk: which section it came from (only this varies per loop)
                "text": chunk,
            }
            results.append(dictionary)

    return results  # flat list of self-describing chunk dicts

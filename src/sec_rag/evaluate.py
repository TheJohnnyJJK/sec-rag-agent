from sec_rag.generate import answer_question

## Judge prompt for eval
judge_rules = """You are an impartial evaluator scoring a RAG system's answers. You compare the system's answer against a reference (correct) answer and decide whether the system's answer is correct.

You will receive:
- QUESTION: the question that was asked
- REFERENCE: the known-correct answer
- ANSWER: the system's answer to evaluate

Judge on MEANING, not wording. The ANSWER does not need to match the REFERENCE word-for-word, or have the same format, length, or level of detail. It is correct if it conveys the same factual information the REFERENCE conveys.

Scoring rules:
- PASS if the ANSWER contains the key facts of the REFERENCE and does not add incorrect or contradictory information. Extra correct detail is fine. Different phrasing is fine.
- FAIL if the ANSWER misses a key fact from the REFERENCE, states something factually different, or contradicts the REFERENCE.

Refusal handling:
- If the REFERENCE is "I don't have enough information", then the ANSWER is a PASS only if it also declines to answer (says it doesn't have the information / cannot find it in the provided material). It is a FAIL if it instead provides a substantive answer.
- If the REFERENCE is a real answer but the ANSWER refuses ("I don't have enough information"), that is a FAIL — it should have answered.

Numbers: if the REFERENCE contains a specific figure, the ANSWER must contain the same figure (allowing for equivalent formatting like "$416,161 million" vs "$416.2 billion") to PASS.

Output format — respond with EXACTLY two lines and nothing else:
VERDICT: PASS or FAIL
REASON: one short sentence explaining the verdict"""

judge_rules_agent = """You are grading whether an AI answer matches the expected answer.

You are NOT grounded in any document. You are NOT checking sources or citations. Your only job is to compare two answers and decide if they convey the same key facts.

You will receive:
- The QUESTION that was asked
- The EXPECTED answer (the correct answer)
- The SUBMITTED answer (the AI's answer to grade)

Grade PASS if the submitted answer contains the same key facts and figures as the expected answer — the core numbers and conclusion match. Minor differences in wording, formatting, rounding, or extra explanation are fine. What matters is whether the essential answer is correct.

Grade FAIL if the submitted answer gives a different number, a wrong conclusion, refuses to answer, or is missing the key fact.

Respond with exactly one word: PASS or FAIL."""


## Eval source of truth questions to be compared to the llm output
eval_set = [
    # ===== MESSY 10 (vague / casual real-user phrasing) =====
    {"question": "what does apple even make",
     "expected": "Smartphones, personal computers, tablets, wearables and accessories, plus related services."},
    {"question": "which iphones are out",
     "expected": "iPhone 17 Pro, iPhone Air, iPhone 17, iPhone 16, and iPhone 16e."},
    {"question": "how many people work at apple",
     "expected": "Approximately 166,000 full-time equivalent employees."},
    {"question": "what does apple compete on",
     "expected": "Price; product and service features including security; relative price and performance; product and service quality and reliability; design and technology innovation; a strong third-party software and accessories ecosystem; marketing and distribution capability; service and support; corporate reputation; and the ability to protect and enforce its intellectual property."},
    {"question": "apple revenue",
     "expected": "$416,161 million (about $416.2 billion)."},
    {"question": "how much profit did they make",
     "expected": "$112,010 million."},
    {"question": "iphone sales",
     "expected": "$209,586 million."},
    {"question": "how much on r&d",
     "expected": "$34,550 million."},
    {"question": "how much cash do they have",
     "expected": "$132.4 billion."},
    {"question": "how much stock did they buy back",
     "expected": "$89.3 billion (402 million shares)."},

    # ===== CLEAN 10 (kept formal; includes the refusals) =====
    {"question": "Where is Apple's headquarters located?",
     "expected": "Cupertino, California."},
    {"question": "What are Apple's reportable segments?",
     "expected": "Americas, Europe, Greater China, Japan, and Rest of Asia Pacific."},
    {"question": "On which stock exchange is Apple's common stock traded, and under what symbol?",
     "expected": "The Nasdaq Stock Market LLC, under the symbol AAPL."},
    {"question": "How many shareholders of record did Apple have as of October 17, 2025?",
     "expected": "22,429 shareholders of record."},
    {"question": "Who is Apple's independent registered public accounting firm?",
     "expected": "Ernst & Young LLP."},
    {"question": "What was Apple's Services net sales in 2025?",
     "expected": "$109,158 million."},
    {"question": "What was Apple's effective tax rate in 2025?",
     "expected": "15.6%."},
    {"question": "What is Apple's fiscal year?",
     "expected": "The 52- or 53-week period that ends on the last Saturday of September."},
    {"question": "What is the exact salary of Apple's CEO?",
     "expected": "I don't have enough information"},
    {"question": "What were Apple's total net sales in 2019?",
     "expected": "I don't have enough information"},
]


## Multi-step eval for the agent (rag.run_agent). Each question needs two or more
## lookups plus arithmetic, so a single retrieve→answer pass can't solve it — the
## agent has to chain search_filings + calculator. expected_steps is the intended
## reasoning path (recorded for inspection, not asserted); the judge scores the answer.
agent_eval = [
    {
        "question": "What was Apple's R&D spending as a percentage of revenue in 2025?",
        "expected_answer": "About 8.3% ($34,550M R&D / $416,161M revenue)",
        "expected_steps": ["look up R&D spending", "look up total revenue", "compute ratio"],
    },
    {
        "question": "How much did Apple's net income change from 2024 to 2025?",
        "expected_answer": "Up about $18,274M (from $93,736M to $112,010M)",
        "expected_steps": ["look up 2025 net income", "look up 2024 net income", "compute difference"],
    },
    {
        "question": "Which of Apple's operating expense lines was larger in 2025, R&D or SG&A, and by how much?",
        "expected_answer": "R&D was larger by about $6,949M ($34,550M vs $27,601M)",
        "expected_steps": ["look up R&D", "look up SG&A", "compare / subtract"],
    },
    {
        "question": "What was Apple's total operating expense in 2025, combining R&D and SG&A?",
        "expected_answer": "About $62,151M ($34,550M + $27,601M)",
        "expected_steps": ["look up R&D", "look up SG&A", "sum them"],
    },
    {
        "question": "How much did Apple's R&D spending grow from 2023 to 2025 in absolute terms?",
        "expected_answer": "Up about $4,635M (from $29,915M to $34,550M)",
        "expected_steps": ["look up 2025 R&D", "look up 2023 R&D", "compute difference"],
    },
]


def judge_the_eval(reference_answer, llm_answer, question, client, judge_rules):
    # Build the judge prompt: the known-correct answer, the pipeline's answer, and the question
    prompt = f"""
        <reference>
        {reference_answer}
        </reference>

        <llm_answer>
        {llm_answer}
        </llm_answer>

        Question: {question}"""

    # Send to Claude with the judge rubric as the system prompt; returns PASS/FAIL + reason
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=judge_rules,
        messages=[{"role": "user", "content": prompt}],
    )

    # Hand back just the verdict text, not the whole response object
    return response.content[0].text


def run_eval(eval_set, vectorestore, bm25, fixed_chunks, reranker, client, rules, judge_rules, rerank_top_n=10):
    scored = []  # scorecard: one result dict per eval question

    for item in eval_set:
        # Run the full RAG pipeline on this question to get an answer
        llm_answer = answer_question(item["question"], vectorestore, bm25, fixed_chunks, reranker, client, rules, rerank_top_n)

        # Judge that answer against the known-correct one (returns verdict text directly)
        verdict_eval = judge_the_eval(item["expected"], llm_answer, item["question"], client, judge_rules)

        # Save the result, with a True/False for whether the judge said PASS
        scored.append({
            "question": item["question"],
            "verdict": verdict_eval,
            "passed": "PASS" in verdict_eval,
        })

    # Tally: True counts as 1, so summing the "passed" flags gives the number correct
    passes = sum(s["passed"] for s in scored)
    print(f"Score: {passes}/{len(scored)}")

    return scored


def run_agent_eval(agent_eval, rag, judge_rules_agent):
    """Run each multi-step question through ``rag.run_agent``, judge the answer,
    and print a per-question + summary report.

    Calls the existing pieces (``rag.run_agent``, ``judge_the_eval``) — nothing
    is reimplemented here. Returns one result dict per question with the tool
    path actually taken and the token cost.
    """
    results = []
    for item in agent_eval:
        # Full agentic loop: the agent chains search_filings + calculator itself.
        answer, steps, usage = rag.run_agent(item["question"])
        tools_used = [s["tool"] for s in steps]          # the actual path

        # Same judge function as the main eval, with the terse agent rubric.
        verdict = judge_the_eval(item["expected_answer"], answer, item["question"], rag.client, judge_rules_agent)

        results.append({
            "question": item["question"],
            "answer": answer,
            "correct": verdict,
            "expected_steps": item["expected_steps"],
            "tools_used": tools_used,
            "num_steps": len(steps),
            "tokens": usage["input_tokens"] + usage["output_tokens"],
        })

    # --- Per-question report ---
    for i, r in enumerate(results, 1):
        passed = "PASS" in r["correct"]
        flag = "PASS" if passed else "FAIL"
        path = " → ".join(r["tools_used"]) or "(no tools)"
        print(f"[{flag}] Q{i}: {r['question']}")
        print(f"       tools: {path}  ·  {r['num_steps']} steps  ·  {r['tokens']} tokens")
        print(f"       answer: {r['answer'][:160].strip()}")
        if not passed:
            print(f"       verdict: {r['correct'].strip()}")

    # --- Summary ---
    passes = sum("PASS" in r["correct"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    print(f"\nAgent eval: {passes}/{len(results)} passed  ·  {total_tokens} tokens total")

    return results

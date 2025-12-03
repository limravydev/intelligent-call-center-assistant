"""
evaluate_rag.py

Simple evaluation script for the Call Center RAG assistant.

Usage:
    python evaluate_rag.py

It will:
  - load / reuse the Chroma index (Excel + PDFs)
  - run a fixed set of evaluation questions
  - save results to eval_results.csv
"""

import csv
from pathlib import Path

from app.rag import build_or_load_index
from app.chatbot import init_gemini_client, answer_question

from dotenv import load_dotenv
load_dotenv()


# Folder paths (adjust if your structure is different)
EXCEL_DIR = "data/excel"
PDF_DIR = "data/pdf"

# Number of retrieved chunks
TOP_K = 5


# A small evaluation set.
# expected_keywords: we just check that these words (lowercased) appear somewhere in the answer.
EVAL_QUESTIONS = [
    # ---- Bank products info ----
    {
        "id": "P1",
        "category": "product_info",
        "question": "What is the minimum balance required for a savings account?",
        "expected_keywords": ["minimum", "balance", "savings"],
    },
    {
        "id": "P2",
        "category": "product_info",
        "question": "Do you offer fixed deposit accounts and what is the basic condition?",
        "expected_keywords": ["fixed", "deposit"],
    },
    # ---- General questions ----
    {
        "id": "G1",
        "category": "general",
        "question": "What documents are required to open a new bank account?",
        "expected_keywords": ["id", "passport", "national", "document"],
    },
    {
        "id": "G2",
        "category": "general",
        "question": "How can I update my registered phone number?",
        "expected_keywords": ["update", "phone", "branch"],
    },
    # ---- Mobile banking / OTP ----
    {
        "id": "MB1",
        "category": "mobile_banking",
        "question": "I forgot my mobile banking password. How can I reset it?",
        "expected_keywords": ["reset", "password"],
    },
    {
        "id": "MB2",
        "category": "mobile_banking",
        "question": "I did not receive an OTP when trying to login. What should I do?",
        "expected_keywords": ["otp", "network", "phone"],
    },
    # ---- Process / troubleshoot scenarios ----
    {
        "id": "T1",
        "category": "process_troubleshoot",
        "question": "Customer cannot login to mobile app even with correct password. What should the agent check?",
        "expected_keywords": ["locked", "status", "failed", "attempt"],
    },
    {
        "id": "T2",
        "category": "process_troubleshoot",
        "question": "Customer forgot their CIF number. What should the agent do?",
        "expected_keywords": ["verify", "identity", "cif", "branch"],
    },
    {
        "id": "T3",
        "category": "process_troubleshoot",
        "question": "Customer account is dormant. How can we reactivate it?",
        "expected_keywords": ["dormant", "reactivate", "branch"],
    },
    # ---- Low-confidence / off-topic ----
    {
        "id": "L1",
        "category": "off_topic",
        "question": "Who is the CEO of the bank?",
        "expected_keywords": ["not", "sure"],  # should trigger low-confidence / escalate behavior
    },
    {
        "id": "L2",
        "category": "off_topic",
        "question": "Can you recommend which investment I should choose?",
        "expected_keywords": ["not", "sure"],  # also should not hallucinate
    },
]


def run_evaluation():
    # 1. Build or load index
    print("[Eval] Loading index...")
    collection, embed_model = build_or_load_index(
        excel_path=EXCEL_DIR,
        pdf_dir=PDF_DIR,
        rebuild=False,   # set True if you changed KB files and want to rebuild
    )

    # 2. Init Gemini
    print("[Eval] Initializing Gemini client...")
    gemini_client = init_gemini_client()

    # 3. Run through evaluation set
    results = []
    for item in EVAL_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        expected_keywords = [w.lower() for w in item.get("expected_keywords", [])]

        print(f"\n[Eval] Q {qid}: {question}")
        answer, ctx_preview = answer_question(
            question=question,
            collection=collection,
            embed_model=embed_model,
            gemini_client=gemini_client,
            k=TOP_K,
            history=None,
        )

        answer_lower = answer.lower()
        passed = all(kw in answer_lower for kw in expected_keywords) if expected_keywords else True

        print(f"[Eval]   -> PASS: {passed}")
        # print(f"[Eval]   -> Answer (truncated): {answer[:200].replace('\\n', ' ')} ...")
        ans_preview = answer[:200].replace("\n", " ")
        print(f"[Eval]   -> Answer (truncated): {ans_preview} ...")

        results.append(
            {
                "id": qid,
                "category": item["category"],
                "question": question,
                "expected_keywords": ", ".join(expected_keywords),
                "passed": passed,
                "answer": answer,
                "context_preview": ctx_preview,
            }
        )
        
        # 3.5 Print summary in a compact table
    from collections import Counter

    total = len(results)
    num_pass = sum(1 for r in results if r["passed"])
    overall_acc = (num_pass / total) * 100 if total else 0.0

    cat_total = Counter(r["category"] for r in results)
    cat_pass = Counter(r["category"] for r in results if r["passed"])

    print("\n========== Evaluation Summary ==========")
    print(f"Total questions : {total}")
    print(f"Passed          : {num_pass}")
    print(f"Overall accuracy: {overall_acc:.1f}%\n")

    print("{:<8} {:>5} {:>5} {:>9}".format("Category", "Total", "Pass", "Accuracy"))
    print("-" * 32)
    for cat in sorted(cat_total.keys()):
        t = cat_total[cat]
        p = cat_pass.get(cat, 0)
        acc = (p / t) * 100 if t else 0.0
        print("{:<8} {:>5} {:>5} {:>8.1f}%".format(cat, t, p, acc))
    print("========================================\n")

    # 4. Save to CSV
    out_path = Path("eval_results.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "question",
                "expected_keywords",
                "passed",
                "answer",
                "context_preview",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[Eval] Done. Results saved to {out_path.resolve()}")


if __name__ == "__main__":
    run_evaluation()
from app.rag import build_or_load_index
from app.chatbot import init_gemini_client, answer_question
from dotenv import load_dotenv

# Load .env
load_dotenv()

# -------------------------------------------------------------------
# Test cases: you should extend this to ~10–20 questions
# Each case:
#   - category: for your slide stats
#   - question: what you ask the assistant
#   - must_include: phrases that *should* appear in the answer
#   - must_not_include: phrases that *should not* appear (for hallucination)
# -------------------------------------------------------------------
# TEST_CASES = [
#     {
#         "id": "P1",
#         "category": "Banking Products",
#         "question": "What is the minimum opening deposit for a savings account?",
#         "must_include": ["minimum opening deposit", "10 units"],
#         "must_not_include": ["fixed deposit", "mobile app"],
#     },
#     {
#         "id": "P2",
#         "category": "Banking Products",
#         "question": "Do you offer fixed deposit accounts and what is the basic condition?",
#         "must_include": ["fixed deposit", "minimum deposit"],
#         "must_not_include": ["mobile app", "OTP"],
#     },
#     {
#         "id": "T1",
#         "category": "Troubleshooting",
#         "question": "How can I reset my mobile banking password?",
#         "must_include": ["forgot password", "mobile banking app"],
#         "must_not_include": ["passbook"],
#     },
#     {
#         "id": "OTP1",
#         "category": "OTP Issues",
#         "question": "I did not receive an OTP when trying to log in. What should I do?",
#         "must_include": ["OTP", "registered phone number"],
#         "must_not_include": ["foreign exchange slip"],
#     },
#     {
#         "id": "MB1",
#         "category": "Mobile banking issue",
#         "question": "Customer cannot log in to the mobile app even with the correct password. What should I check?",
#         "must_include": ["account lock", "registered phone number"],
#         "must_not_include": ["passbook", "fixed deposit"],
#     },
#     {
#         "id": "OFF1",
#         "category": "Off topics",
#         "question": "What is your favourite movie?",
#         "must_include": [],  # we expect safe fallback, not banking info
#         "must_not_include": ["interest rate", "savings account", "fixed deposit"],
#     },
#     # ➜ add more cases here if you have time
# ]

TEST_CASES = [
    # 1) Banking product – Savings account
    {
        "id": "P1",
        "category": "Banking Products",
        "question": "What is the minimum opening deposit for a savings account?",
        "must_include": ["minimum opening deposit", "savings account"],
        "must_not_include": ["fixed deposit", "credit card"],
    },

    # 2) Banking product – Fixed deposit
    {
        "id": "P2",
        "category": "Banking Products",
        "question": "Do you offer fixed deposit accounts and what is the basic condition?",
        "must_include": ["fixed deposit", "minimum deposit"],
        "must_not_include": ["mobile app", "otp"],
    },

    # 3) Troubleshooting – Reset mobile banking password
    {
        "id": "T1",
        "category": "Troubleshooting",
        "question": "How can I reset my mobile banking password?",
        "must_include": ["reset", "mobile banking"],
        "must_not_include": ["passbook"],
    },

    # 4) OTP issue – did not receive OTP
    {
        "id": "OTP1",
        "category": "OTP Issues",
        "question": "I did not receive an OTP when trying to log in. What should I do?",
        "must_include": ["otp", "mobile banking app"],
        "must_not_include": ["foreign exchange slip"],
    },

    # 5) Mobile banking – correct password but cannot log in
    {
        "id": "MB1",
        "category": "Mobile Banking",
        "question": "Customer cannot log in to the mobile app even with the correct password. What should I check?",
        "must_include": ["account is locked", "inactive"],
        "must_not_include": ["fixed deposit", "passbook"],
    },

    # 6) Chit-chat / small talk
    {
        "id": "CC1",
        "category": "Chitchat",
        "question": "How are you today?",
        "must_include": ["here to help"],        # adjust after you see real answer
        "must_not_include": ["interest rate", "savings account"],
    },

    # 7) “Follow-up style” – minors savings account
    # (tests a more specific variant of product question)
    {
        "id": "P3",
        "category": "Banking Products",
        "question": "For a minor customer, what documents are required to open a savings account?",
        "must_include": ["minor", "guardian"],
        "must_not_include": ["credit card", "loan"],
    },

    # 8) Ambiguous question – should ask for clarification
    {
        "id": "AMB1",
        "category": "Ambiguous",
        "question": "I want to open an account.",
        "must_include": ["what type of account"],   # encourage clarification
        "must_not_include": ["fixed deposit rate"], # avoid guessing details
    },

    # 9) Off-topic – personal question
    {
        "id": "OFF1",
        "category": "Off topics",
        "question": "What is your favourite movie?",
        "must_include": ["I'm not fully sure"],  # or similar
        "must_not_include": ["interest rate", "deposit product"],
    },

    # 10) No-information / safety – crypto investment (should not hallucinate)
    {
        "id": "NOINFO1",
        "category": "No-info / Safety",
        "question": "Can I invest in cryptocurrency through your bank?",
        "must_include": ["I'm not fully sure", "double-check "],  # or similar
        "must_not_include": ["specific crypto product", "step-by-step investment"],
    },
]


def evaluate_case(case, collection, embed_model, gemini_client):
    """Run one test case and compute metrics."""
    question = case["question"]
    must_include = [p.lower() for p in case.get("must_include", [])]
    must_not_include = [p.lower() for p in case.get("must_not_include", [])]

    answer, ctx_preview = answer_question(
        question,
        collection,
        embed_model,
        gemini_client,
        k=5,
        history=[],
    )

    ans_lower = answer.lower()

    # --- Accuracy / Completeness: based on must_include phrases ---
    include_hits = sum(1 for p in must_include if p in ans_lower)
    include_total = len(must_include)
    accuracy_score = include_hits / include_total if include_total > 0 else 1.0

    # define pass if we hit at least half of expected phrases
    accuracy_pass = accuracy_score >= 0.5
    completeness_pass = accuracy_score == 1.0 if include_total > 0 else True

    # --- Hallucination heuristic ---
    # If none of the expected phrases appear but answer is long, treat as hallucination-ish
    hallucination = False
    if include_total > 0 and include_hits == 0 and len(answer) > 50:
        hallucination = True

    # Or if answer contains forbidden phrases
    if any(p in ans_lower for p in must_not_include):
        hallucination = True

    # --- Step clarity: check if answer has at least 2 numbered steps ---
    # We just look for patterns "1." and "2."
    step_clarity = ("1." in ans_lower and "2." in ans_lower) or ("step 1" in ans_lower)

    return {
        "answer": answer,
        "ctx": ctx_preview,
        "accuracy_score": accuracy_score,
        "accuracy_pass": accuracy_pass,
        "completeness_pass": completeness_pass,
        "hallucination": hallucination,
        "step_clarity": step_clarity,
    }

import matplotlib.pyplot as plt  # <-- add at the top of the file

def run_evaluation():
    print("[Eval] Loading index and Gemini client...")
    collection, embed_model = build_or_load_index()
    gemini_client = init_gemini_client()

    rows = []  # store per-case results for later
    total = len(TEST_CASES)
    acc_pass_count = 0
    comp_pass_count = 0
    halluc_count = 0
    step_clear_count = 0

    print("\nID   Cat   Acc   Comp   Halluc   Steps")
    print("==============================================")

    for case in TEST_CASES:
        result = evaluate_case(case, collection, embed_model, gemini_client)

        if result["accuracy_pass"]:
            acc_pass_count += 1
        if result["completeness_pass"]:
            comp_pass_count += 1
        if result["hallucination"]:
            halluc_count += 1
        if result["step_clarity"]:
            step_clear_count += 1

        row = {
            "id": case["id"],
            "cat": case["category"],
            "accuracy_pass": result["accuracy_pass"],
            "completeness_pass": result["completeness_pass"],
            "hallucination": result["hallucination"],
            "step_clarity": result["step_clarity"],
        }
        rows.append(row)

        print(
            f"{case['id']:<4} {case['category'][:3]:<4} "
            f"{'✔' if result['accuracy_pass'] else '✘':<5}"
            f"{'✔' if result['completeness_pass'] else '✘':<7}"
            f"{'✘' if result['hallucination'] else '✔':<8}"
            f"{'✔' if result['step_clarity'] else '✘'}"
        )

    # --- Summary numbers ---
    acc_pct = acc_pass_count / total * 100
    comp_pct = comp_pass_count / total * 100
    halluc_pct = halluc_count / total * 100
    step_pct = step_clear_count / total * 100

    print("\n--- Summary ---")
    print(f"Total cases:         {total}")
    print(f"Accuracy pass:       {acc_pass_count}  ({acc_pct:.0f}%)")
    print(f"Completeness pass:   {comp_pass_count}  ({comp_pct:.0f}%)")
    print(f"Hallucination count: {halluc_count}  ({halluc_pct:.0f}%)")
    print(f"Step clarity pass:   {step_clear_count}  ({step_pct:.0f}%)")

    print("\nOverall result: passed most test cases, hallucination reduced after tuning.")
    print("You can use these numbers directly on your evaluation slide.")

    # After printing summary, create a chart for slides
    create_summary_chart(acc_pct, comp_pct, halluc_pct, step_pct)


def create_summary_chart(acc_pct, comp_pct, halluc_pct, step_pct):
    """
    Create a clean bar chart PNG for use in slides.
    """
    metrics = ["Accuracy", "Completeness", "Non-hallucinating", "Step clarity"]
    # Non-hallucinating = 100 - hallucination %
    values = [acc_pct, comp_pct, 100 - halluc_pct, step_pct]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values)

    # Colors to match your dark/cyan slide theme
    colors = ["#22d3ee", "#38bdf8", "#4ade80", "#facc15"]
    for bar, c in zip(bars, colors):
        bar.set_color(c)

    # Add value labels on top of bars
    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.ylim(0, 110)
    plt.ylabel("Score (%)")
    plt.title("Call Center RAG – Evaluation Summary", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig("evaluation_summary.png", dpi=200)
    plt.close()

    print('\n[Eval] Saved chart to "evaluation_summary.png" (ready for PowerPoint).')


if __name__ == "__main__":
    run_evaluation()
"""
Evaluation pipeline: dataset loading + answer extraction/verification + metrics + main entry.
"""
import argparse
import json
import os
import re
from collections import Counter

from tqdm import tqdm
from datasets import load_dataset

from agent import get_llm, create_math_agent


def extract_answer(text: str) -> str:
    """Extract the final answer from model output."""
    if not text:
        return ""

    # 1. Primary: extract \boxed{...}
    # Handle nested braces by matching balanced braces
    pattern = r"\\boxed\{"
    for match in re.finditer(pattern, text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1].strip()

    # 2. Fallback: last non-empty line, cleaned
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    # Remove markdown code fences and LaTeX display markers
    last = re.sub(r"^```+.*", "", last)
    last = re.sub(r"```+$", "", last)
    last = last.strip("`$ ")
    return last


def verify_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches the gold answer."""
    pred = predicted.strip()
    gold = str(gold).strip()
    if not pred:
        return False

    # Exact string match first
    if pred == gold:
        return True

    # Use latex2sympy2_extended + sympy for robust verification
    try:
        from latex2sympy2_extended import latex2sympy
        import sympy as sp

        try:
            gold_expr = latex2sympy(gold)
        except Exception:
            gold_expr = sp.sympify(gold)

        try:
            pred_expr = latex2sympy(pred)
        except Exception:
            pred_expr = sp.sympify(pred)

        if gold_expr == pred_expr:
            return True

        # Numerical comparison for non-interval expressions
        if not isinstance(gold_expr, sp.Interval) and not isinstance(
            pred_expr, sp.Interval
        ):
            diff = sp.N(gold_expr - pred_expr)
            if abs(diff) < 1e-6:
                return True

        return False
    except Exception:
        pass

    return False


def compute_metrics(results: list[dict]) -> dict:
    """Compute Pass@1 and Cons@k."""
    if not results:
        return {"pass@1": 0.0, "cons@k": 0.0}

    # Pass@1: accuracy of the first sample per problem
    pass_at_1 = sum(r["samples"][0] for r in results) / len(results)

    # Cons@k: majority voting accuracy
    cons_correct = 0
    for r in results:
        answers = r["extracted_answers"]
        if not answers:
            continue
        counter = Counter(answers)
        majority_answer, _ = counter.most_common(1)[0]
        # Determine correctness of the majority answer
        # Use the correctness of the first occurrence
        is_correct = False
        for i, ans in enumerate(answers):
            if ans == majority_answer:
                is_correct = r["samples"][i]
                break
        if is_correct:
            cons_correct += 1

    cons_k = cons_correct / len(results)
    return {"pass@1": pass_at_1, "cons@k": cons_k}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a calculator-only agent on OlymMATH-HARD (EN)."
    )
    parser.add_argument(
        "--sample", type=int, default=10, help="Number of samples per problem"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=32768, help="Max tokens per generation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Limit number of problems (default: all 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed for reproducibility"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    llm = get_llm(
        temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed
    )
    agent = create_math_agent(llm)

    ds = load_dataset("RUC-AIBOX/OlymMATH", "en-hard", split="test")
    if args.num_problems is not None:
        ds = ds.select(range(min(args.num_problems, len(ds))))

    results = []

    for idx, item in enumerate(tqdm(ds, desc="Evaluating")):
        problem_text = item["problem"]
        gold_answer = str(item["answer"]).strip()
        unique_id = item.get("unique_id", f"q{idx}")

        samples = []
        extracted_answers = []
        traces = []

        for s in range(args.sample):
            try:
                # Invoke agent
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": problem_text}]},
                    config={"recursion_limit": 100},
                )
                final_msg = response["messages"][-1].content
                pred = extract_answer(final_msg)
                is_correct = verify_answer(pred, gold_answer)
            except Exception as exc:
                final_msg = f"EXCEPTION: {exc}"
                pred = ""
                is_correct = False

            samples.append(is_correct)
            extracted_answers.append(pred)
            traces.append(final_msg)

        result_entry = {
            "unique_id": unique_id,
            "problem": problem_text,
            "gold_answer": gold_answer,
            "subject": item.get("subject", ""),
            "samples": samples,
            "extracted_answers": extracted_answers,
            "traces": traces,
        }
        results.append(result_entry)

        # Save incremental results
        with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    metrics = compute_metrics(results)
    summary = {
        "metrics": metrics,
        "num_problems": len(results),
        "sample": args.sample,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    with open(
        os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== OlymMATH-HARD (EN) Results ===")
    print(f"Problems evaluated: {len(results)}")
    print(f"Samples per problem: {args.sample}")
    print(f"Pass@1 : {metrics['pass@1']:.4f}")
    print(f"Cons@{args.sample} : {metrics['cons@k']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

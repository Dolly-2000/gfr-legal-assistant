"""
GFR RAG Evaluation Script — 4-bit Model
=========================================
Same evaluation as evaluate.py but uses 4-bit quantized Qwen2.5-14B-Instruct.
Also supports --model 7b flag to evaluate the smaller 7B model.

Usage:
  python src/evaluate_4bit.py                    # Retrieval-only (no GPU)
  python src/evaluate_4bit.py --full             # Full eval with 4-bit 14B
  python src/evaluate_4bit.py --full --model 7b  # Full eval with 4-bit 7B
  python src/evaluate_4bit.py --full --export results_4bit
"""

import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import json
import sys
import shutil
import argparse
import time
import csv
from collections import defaultdict

# Import ground truth from main evaluate script
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import GROUND_TRUTH, load_vectorstore, evaluate_retrieval


def evaluate_full_4bit(vectorstore, embeddings, model_size="14b"):
    """Full evaluation with 4-bit quantized LLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    import re

    if model_size == "7b":
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        model_label = "Qwen2.5-7B-Instruct (4-bit NF4)"
    else:
        model_id = "Qwen/Qwen2.5-14B-Instruct"
        model_label = "Qwen2.5-14B-Instruct (4-bit NF4)"

    print("=" * 70)
    print(f"FULL RAG EVALUATION — {model_label}")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU free memory: {free_mem:.1f} GB")
    else:
        print("ERROR: No GPU available. 4-bit model requires CUDA.")
        return None

    # MMR retriever (same as app_4bit.py)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
    )

    print(f"Loading LLM: {model_id} (4-bit NF4)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=1024, temperature=0.2, do_sample=True,
        repetition_penalty=1.1, return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    template = """<|im_start|>system
You are a legal and financial expert on the General Financial Rules (GFR) 2025 of India.
Answer the user's question using ONLY the provided context below.
Important instructions:
- Extract and include ALL relevant definitions, provisions, and details from the context.
- If definitions or information are spread across multiple chunks, combine them into one complete answer.
- Always cite the specific Rule numbers (e.g., "As per Rule 2(vi)...").
- Structure your answer clearly with bullet points or numbered lists when listing multiple items.
- If the answer is not in the context, say so explicitly.

Context:
{context}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(
            f"Rule {doc.metadata.get('rule_number', 'N/A')} ({doc.metadata.get('title', 'N/A')}):\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Run evaluation
    gen_results = []
    total_start = time.time()
    
    for i, gt in enumerate(GROUND_TRUTH):
        print(f"\n  Q{i+1:02d}/{len(GROUND_TRUTH)}: {gt['question'][:60]}...")
        start = time.time()

        try:
            response = rag_chain.invoke(gt["question"])
            elapsed = time.time() - start

            answer = response.strip()

            # Faithfulness: does the answer mention expected rule numbers?
            rules_mentioned = re.findall(r'Rule\s+(\d+)', answer)
            rule_overlap = len(set(rules_mentioned) & set(gt["expected_rules"]))
            faithfulness = rule_overlap / len(gt["expected_rules"]) if gt["expected_rules"] else 0

            # Relevancy: answer length > 50 chars and doesn't say "cannot find"
            is_relevant = len(answer) > 50 and "cannot find" not in answer.lower()

            gen_results.append({
                "question": gt["question"],
                "expected_rules": gt["expected_rules"],
                "generated_answer": answer,
                "answer_length": len(answer),
                "time_sec": elapsed,
                "faithfulness": faithfulness,
                "is_relevant": is_relevant,
            })
            print(f"       Time: {elapsed:.1f}s | Faithfulness: {faithfulness:.2f} | Relevant: {is_relevant} | Len: {len(answer)}")
            print(f"       Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        except Exception as e:
            print(f"       ERROR: {e}")
            gen_results.append({
                "question": gt["question"],
                "expected_rules": gt["expected_rules"],
                "generated_answer": f"ERROR: {e}",
                "answer_length": 0, "time_sec": 0,
                "faithfulness": 0, "is_relevant": False,
            })

    total_elapsed = time.time() - total_start

    # Aggregate
    avg_faith = sum(r["faithfulness"] for r in gen_results) / len(gen_results)
    relevancy = sum(r["is_relevant"] for r in gen_results) / len(gen_results)
    avg_time = sum(r["time_sec"] for r in gen_results) / len(gen_results)

    print(f"\n{'─' * 50}")
    print(f"  Model:               {model_label}")
    print(f"  Avg Faithfulness:    {avg_faith:.2%}")
    print(f"  Answer Relevancy:    {relevancy:.2%}")
    print(f"  Avg Response Time:   {avg_time:.1f}s")
    print(f"  Total Time:          {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'─' * 50}\n")

    return {"model": model_label, "faithfulness": avg_faith, "relevancy": relevancy, "avg_time": avg_time, "total_time": total_elapsed, "details": gen_results}


def export_results(retrieval_results, gen_results, prefix):
    """Export evaluation results to CSV."""
    ret_file = f"{prefix}_retrieval.csv"
    with open(ret_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected", "retrieved", "hit", "rr", "precision"])
        writer.writeheader()
        for r in retrieval_results["details"]:
            writer.writerow({
                "question": r["question"],
                "expected": "|".join(r["expected"]),
                "retrieved": "|".join(r["retrieved"]),
                "hit": r["hit"],
                "rr": f"{r['rr']:.4f}",
                "precision": f"{r['precision']:.4f}",
            })
    print(f"Saved: {ret_file}")

    if gen_results:
        gen_file = f"{prefix}_generation.csv"
        with open(gen_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "expected_rules", "generated_answer", "answer_length", "time_sec", "faithfulness", "is_relevant"])
            writer.writeheader()
            for r in gen_results["details"]:
                writer.writerow(r)
        print(f"Saved: {gen_file}")


def main():
    parser = argparse.ArgumentParser(description="GFR RAG Evaluation (4-bit)")
    parser.add_argument("--full", action="store_true", help="Run full evaluation with 4-bit LLM (needs GPU)")
    parser.add_argument("--model", type=str, default="14b", choices=["14b", "7b"], help="Model size: 14b or 7b (default: 14b)")
    parser.add_argument("--export", type=str, default=None, help="Export results to CSV with given prefix")
    parser.add_argument("--k", type=int, default=8, help="Number of documents to retrieve (default: 8)")
    args = parser.parse_args()

    vectorstore, embeddings = load_vectorstore()

    # Retrieval evaluation (same for all models — only embeddings matter)
    retrieval_results = evaluate_retrieval(vectorstore, k=args.k, label=f"MMR k={args.k}")

    # Full evaluation with generation
    gen_results = None
    if args.full:
        gen_results = evaluate_full_4bit(vectorstore, embeddings, model_size=args.model)

    # Export
    if args.export:
        export_results(retrieval_results, gen_results, args.export)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Questions:          {len(GROUND_TRUTH)}")
    print(f"  Hit Rate@{args.k}:       {retrieval_results['hit_rate']:.2%}")
    print(f"  MRR@{args.k}:            {retrieval_results['mrr']:.4f}")
    print(f"  Avg Precision@{args.k}:  {retrieval_results['avg_precision']:.4f}")
    if gen_results:
        print(f"  Model:              {gen_results['model']}")
        print(f"  Faithfulness:       {gen_results['faithfulness']:.2%}")
        print(f"  Answer Relevancy:   {gen_results['relevancy']:.2%}")
        print(f"  Avg Response Time:  {gen_results['avg_time']:.1f}s")
        print(f"  Total Eval Time:    {gen_results['total_time']:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

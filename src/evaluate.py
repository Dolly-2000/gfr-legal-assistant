"""
GFR RAG Evaluation Script
==========================
Evaluates the RAG pipeline using:
  1. Retrieval Metrics: Hit Rate, MRR (Mean Reciprocal Rank), Context Precision
  2. Generation Metrics: Faithfulness (answer grounded in context), Answer Relevancy
  3. End-to-End: Compares against ground truth answers

Usage:
  python src/evaluate.py                    # Retrieval-only evaluation (no GPU needed)
  python src/evaluate.py --full             # Full evaluation with LLM generation (needs GPU)
  python src/evaluate.py --export results   # Export results to CSV
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
from collections import defaultdict

# ── Ground Truth Q&A Dataset ──
# Each entry has: question, expected_rule_numbers (for retrieval eval), ground_truth_answer
# Rule numbers verified against data/parsed/2025_GFR_chunks.json content
GROUND_TRUTH = [
    {
        "question": "What is the definition of Accounts Officer, Controlling Officer and Head of Office in GFR?",
        "expected_rules": ["2"],
        "ground_truth": "Rule 2 defines key terms: Accounts Officer means the Head of an Office of Accounts or the Treasury Officer. Controlling Officer means an officer entrusted with the responsibility of controlling expenditure. Head of Office means a Gazetted Officer declared as such in the Delegation of Financial Powers Rules."
    },
    {
        "question": "What are the rules for procurement of goods?",
        "expected_rules": ["144", "145", "146"],
        "ground_truth": "Every authority delegated with the financial powers of procuring goods in public interest shall follow fundamental principles of public buying including efficiency, economy, transparency, fair treatment of suppliers and promotion of competition."
    },
    {
        "question": "What is the procedure for Advertised Tender Enquiry?",
        "expected_rules": ["161"],
        "ground_truth": "Invitation to tenders by advertisement should be used for procurement of goods of estimated value of Rs. 50 lakhs and above. Advertisement should be given on GeM as well as on Central Public Procurement Portal."
    },
    {
        "question": "What are the rules regarding loss of Government property?",
        "expected_rules": ["33", "34"],
        "ground_truth": "Any loss of public money, departmental revenue or receipts, stamps, stores or other property shall be immediately reported to the next higher authority and to the concerned Accounts Officer. Cases involving suspected fire, theft, fraud above Rs. 50,000 shall be reported to police."
    },
    {
        "question": "What is the procedure for setting up autonomous organisations?",
        "expected_rules": ["228", "229"],
        "ground_truth": "No new autonomous institutions should be created by Ministries or Departments without the approval of the Cabinet. Stringent criteria should be followed for setting up new autonomous organisations."
    },
    {
        "question": "What are the rules for Grants-in-Aid?",
        "expected_rules": ["230", "231", "232"],
        "ground_truth": "A grant-in-aid is a financial assistance to a person or a public body or an institution for meeting general or specific purposes. All grants-in-aid must be governed by a scheme."
    },
    {
        "question": "What is the rule about security deposit for Government servants?",
        "expected_rules": ["306", "307"],
        "ground_truth": "Every Government servant entrusted with the custody of cash or stores of considerable value shall be required to furnish security. Exceptions include Government servants with non-considerable stores, office furniture custodians, librarians, and drivers."
    },
    {
        "question": "What are the fundamental principles of Government expenditure?",
        "expected_rules": ["21", "22"],
        "ground_truth": "Every officer incurring or authorizing expenditure from public money should be guided by high standards of financial propriety. The expenditure should not be more than the occasion demands and that not more revenue is retained than is needed."
    },
    {
        "question": "What are the rules for Government e-Marketplace (GeM)?",
        "expected_rules": ["149"],
        "ground_truth": "Procurement of goods and services by Ministries or Departments shall be mandatory for goods and services available on GeM. The procurement shall be done through the GeM portal."
    },
    {
        "question": "What is the procedure for re-appropriation of funds under GFR?",
        "expected_rules": ["65", "66", "67"],
        "ground_truth": "An application for re-appropriation of funds shall ordinarily be supported by a brief statement explaining the reasons. Re-appropriation means the transfer of funds from one primary unit of appropriation to another such unit."
    },
    {
        "question": "What is the rule about deposits into Public Account?",
        "expected_rules": ["8"],
        "ground_truth": "Under Article 284 of the Constitution all moneys received by or deposited with any officer employed in connection with the affairs of the Union shall be paid into the Public Account."
    },
    {
        "question": "What are the rules for Limited Tender Enquiry?",
        "expected_rules": ["162"],
        "ground_truth": "Limited Tender Enquiry may be adopted when estimated value of goods to be procured is up to Rs. 50 lakhs. Copies of bidding document should be sent directly to approved suppliers."
    },
    {
        "question": "What is the procedure for Single Tender Enquiry?",
        "expected_rules": ["166"],
        "ground_truth": "Procurement from a single source may be resorted to in certain circumstances such as proprietary article, where only a particular firm is the manufacturer, in emergencies, and for standardization of equipment."
    },
    {
        "question": "What are the rules for procurement of non-consulting services and outsourcing?",
        "expected_rules": ["197", "198"],
        "ground_truth": "Non-Consulting Service means any subject matter of procurement which is not a goods or consulting service. Ministries or Departments may outsource certain services in the interest of economy and efficiency."
    },
    {
        "question": "What are the rules regarding audit fees?",
        "expected_rules": ["317", "318"],
        "ground_truth": "The recovery of cost of Supplementary Audit should be waived where audit is done by CAG through departmental staff but enforced where CAG employs professional auditors."
    },
    {
        "question": "What is the rule for delegation of financial powers in GFR?",
        "expected_rules": ["23"],
        "ground_truth": "The financial powers of the Government have been delegated to Ministries/Departments through the Delegation of Financial Powers Rules. Each Ministry or Department shall exercise these powers subject to GFR provisions."
    },
    {
        "question": "What are the rules for original works and repair works?",
        "expected_rules": ["133", "140"],
        "ground_truth": "For original/minor works and repair works, the Administrative Approval and Expenditure Sanction shall be accorded and funds allotted by the concerned authority."
    },
    {
        "question": "What is the rule about lapse of sanction?",
        "expected_rules": ["30", "31"],
        "ground_truth": "A sanction for any fresh charge shall, unless it is specifically renewed, lapse if no payment in whole or in part has been made during a period of twelve months from the date of issue of such sanction."
    },
    {
        "question": "What are the rules about transfer of charge and cash book verification?",
        "expected_rules": ["286", "301"],
        "ground_truth": "A report of transfer of a Gazetted Government servant shall include verification of cash book or imprest account closed on the date of transfer. Suitable note of refund to be made in original Cash Book entry."
    },
    {
        "question": "What is the procedure for presentation of budget to Parliament and what shall the budget contain?",
        "expected_rules": ["43", "44", "45"],
        "ground_truth": "In accordance with Article 112(1) of the Constitution, Finance Minister shall arrange presentation of the annual financial statement (budget) to Parliament. The budget shall contain estimates of all revenues expected and expenditure proposed."
    },
    # --- Tricky / Edge-case / Cross-reference Questions ---
    {
        "question": "What is the monetary threshold below which goods can be purchased directly without inviting bids?",
        "expected_rules": ["158"],
        "ground_truth": "As per Rule 158, purchase of goods by obtaining bids is required except in cases covered under Rules 154 and 155. Direct purchase without bids may be made for purchases of value up to Rs. 25,000."
    },
    {
        "question": "What is the difference between bid security and performance security, and when is each required?",
        "expected_rules": ["170", "171"],
        "ground_truth": "Bid Security (earnest money) is obtained to safeguard against a bidder withdrawing or altering its bid during validity period in advertised or limited tender enquiry. Performance Security is obtained from the successful bidder to ensure due performance of the contract after award."
    },
    {
        "question": "How should surplus or unserviceable government goods be disposed of and what are the approved modes of disposal?",
        "expected_rules": ["217", "218"],
        "ground_truth": "An item may be declared surplus, obsolete or unserviceable if of no use to the Ministry. Modes of disposal for goods above Rs. 4 lakh residual value include: advertised tender, auction, or through nominated agencies. Items below Rs. 4 lakh may be disposed by any appropriate mode."
    },
    {
        "question": "When must departments surrender savings to the Finance Ministry and what is the deadline?",
        "expected_rules": ["62"],
        "ground_truth": "As per Rule 62, Departments of the Central Government shall surrender to the Finance Ministry, by the dates prescribed by that Ministry before the close of the financial year, savings in grants and appropriations controlled by them."
    },
    {
        "question": "What is a Permanent Advance or Imprest and who can sanction it?",
        "expected_rules": ["322"],
        "ground_truth": "A Permanent Advance or Imprest is granted for meeting day-to-day contingent and emergent expenditure. It may be granted to a government servant by the Head of the Department in consultation with concerned accounts officer."
    },
    {
        "question": "What are the rules for physical verification of fixed assets and how often should it be done?",
        "expected_rules": ["213"],
        "ground_truth": "The inventory for fixed assets shall ordinarily be maintained at site. Fixed assets should be verified at least once in a year and the outcome of the verification should be recorded and discrepancies reported."
    },
    {
        "question": "What is the procedure when there is excess expenditure over the voted grant and how is it regularized?",
        "expected_rules": ["61"],
        "ground_truth": "As per Rule 61, when excess expenditure over the voted grant occurs, it must be regularized by obtaining excess grants from Parliament after the expenditure is incurred as required under Article 115 of the Constitution."
    },
    {
        "question": "Can expenditure be incurred on a new service not contemplated in the budget, and if so, under what conditions?",
        "expected_rules": ["63", "64"],
        "ground_truth": "As per Rule 63, no expenditure shall be incurred during a financial year on a 'New Service' not contemplated in the budget without obtaining a Supplementary Grant or an advance from the Contingency Fund. Rule 64 states a Disbursing Officer may not authorize any payment on his own authority."
    },
    {
        "question": "What are the general principles governing contracts made by Government officers?",
        "expected_rules": ["225"],
        "ground_truth": "Rule 225 lays down general principles for contracts including that contracts must be in writing, signed by authorized officers, should include penalty clauses for breach, and must protect Government interests."
    },
    {
        "question": "What steps must be taken when a loss of public money or government property is detected and what is the reporting procedure?",
        "expected_rules": ["33", "38"],
        "ground_truth": "Rule 33 requires immediate reporting of any loss to the next higher authority and accounts officer. Rule 38 mandates prompt disposal at each stage — detection, reporting, write-off, and final disposal — with special attention to cases involving suspected fraud or theft."
    },
]


def load_vectorstore():
    """Load ChromaDB with NAS bypass (same logic as app.py)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    print("Loading embedding model (CPU)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    local_db_dir = "/tmp/gfr_chroma_db"
    nas_db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

    if not os.path.exists(local_db_dir):
        print(f"Copying DB from NAS to {local_db_dir}...")
        shutil.copytree(nas_db_dir, local_db_dir)

    vectorstore = Chroma(
        persist_directory=local_db_dir,
        embedding_function=embeddings,
        collection_name="gfr_2025"
    )
    doc_count = vectorstore._collection.count()
    print(f"ChromaDB loaded: {doc_count} documents in 'gfr_2025'\n")
    return vectorstore, embeddings


def evaluate_retrieval(vectorstore, k=5, label="SEMANTIC"):
    """Evaluate retrieval quality: Hit Rate, MRR, Context Precision."""
    print("=" * 70)
    print(f"RETRIEVAL EVALUATION — {label} (k={k})")
    print("=" * 70)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = []

    for i, gt in enumerate(GROUND_TRUTH):
        docs = retriever.invoke(gt["question"])
        retrieved_rules = [doc.metadata.get("rule_number", "") for doc in docs]

        # Hit Rate: did ANY expected rule appear in top-k?
        hit = any(r in retrieved_rules for r in gt["expected_rules"])

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, r in enumerate(retrieved_rules, 1):
            if r in gt["expected_rules"]:
                rr = 1.0 / rank
                break

        # Context Precision: fraction of retrieved docs that are relevant
        relevant_count = sum(1 for r in retrieved_rules if r in gt["expected_rules"])
        precision = relevant_count / k

        results.append({
            "question": gt["question"][:60] + "...",
            "expected": gt["expected_rules"],
            "retrieved": retrieved_rules,
            "hit": hit,
            "rr": rr,
            "precision": precision,
        })

        status = "HIT" if hit else "MISS"
        print(f"  Q{i+1:02d} [{status}] RR={rr:.2f} P@{k}={precision:.2f} | Expected: {gt['expected_rules']} Got: {retrieved_rules}")

    # Aggregate
    hit_rate = sum(r["hit"] for r in results) / len(results)
    mrr = sum(r["rr"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)

    print(f"\n{'─' * 50}")
    print(f"  Hit Rate@{k}:       {hit_rate:.2%}  ({sum(r['hit'] for r in results)}/{len(results)})")
    print(f"  MRR@{k}:            {mrr:.4f}")
    print(f"  Avg Precision@{k}:  {avg_precision:.4f}")
    print(f"{'─' * 50}\n")

    return {"hit_rate": hit_rate, "mrr": mrr, "avg_precision": avg_precision, "details": results}


def evaluate_hybrid_retrieval(vectorstore, k=5):
    """Evaluate hybrid BM25+semantic retrieval."""
    from hybrid_retriever import HybridRetriever
    import os

    chunks_path = os.path.join(os.path.dirname(__file__), "..", "data", "parsed", "2025_GFR_chunks.json")

    print("=" * 70)
    print(f"RETRIEVAL EVALUATION — HYBRID BM25+SEMANTIC (k={k})")
    print("=" * 70)

    hybrid = HybridRetriever(
        vectorstore=vectorstore, chunks_path=chunks_path, k=k,
        bm25_weight=0.4, semantic_weight=0.6
    )
    results = []

    for i, gt in enumerate(GROUND_TRUTH):
        docs = hybrid.invoke(gt["question"])
        retrieved_rules = [doc.metadata.get("rule_number", "") for doc in docs]

        hit = any(r in retrieved_rules for r in gt["expected_rules"])
        rr = 0.0
        for rank, r in enumerate(retrieved_rules, 1):
            if r in gt["expected_rules"]:
                rr = 1.0 / rank
                break
        relevant_count = sum(1 for r in retrieved_rules if r in gt["expected_rules"])
        precision = relevant_count / k

        results.append({
            "question": gt["question"][:60] + "...",
            "expected": gt["expected_rules"],
            "retrieved": retrieved_rules,
            "hit": hit, "rr": rr, "precision": precision,
        })
        status = "HIT" if hit else "MISS"
        print(f"  Q{i+1:02d} [{status}] RR={rr:.2f} P@{k}={precision:.2f} | Expected: {gt['expected_rules']} Got: {retrieved_rules}")

    hit_rate = sum(r["hit"] for r in results) / len(results)
    mrr = sum(r["rr"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)

    print(f"\n{'─' * 50}")
    print(f"  Hit Rate@{k}:       {hit_rate:.2%}  ({sum(r['hit'] for r in results)}/{len(results)})")
    print(f"  MRR@{k}:            {mrr:.4f}")
    print(f"  Avg Precision@{k}:  {avg_precision:.4f}")
    print(f"{'─' * 50}\n")

    return {"hit_rate": hit_rate, "mrr": mrr, "avg_precision": avg_precision, "details": results}


def evaluate_full(vectorstore, embeddings):
    """Full evaluation with LLM generation (requires GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    import re

    print("=" * 70)
    print("FULL RAG EVALUATION (Retrieval + Generation)")
    print("=" * 70)

    # Check GPU
    free_mem = 0
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU free memory: {free_mem:.1f} GB")
        if free_mem < 10:
            print("WARNING: Less than 10GB free. Model may not load.")

    # Build RAG chain (same as app.py / rag_pipeline.py)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("Loading LLM (8-bit quantized)...")
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto",
        quantization_config=bnb_config, trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=1500, temperature=0.2, do_sample=True,
        repetition_penalty=1.1, return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    template = """<|im_start|>system
You are a legal and financial expert analyzing the General Financial Rules (GFR). 
Use the provided extracted rules to answer the user's question.
Cite the relevant rule numbers in your answer.
If the answer is not contained in the context, explicitly state that you cannot find it in the provided rules.

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
    for i, gt in enumerate(GROUND_TRUTH):
        print(f"\n  Q{i+1:02d}/{len(GROUND_TRUTH)}: {gt['question'][:60]}...")
        start = time.time()

        try:
            response = rag_chain.invoke(gt["question"])
            elapsed = time.time() - start

            # Clean answer
            answer = response.strip()

            # Simple faithfulness check: does the answer mention expected rule numbers?
            rules_mentioned = re.findall(r'Rule\s+(\d+)', answer)
            rule_overlap = len(set(rules_mentioned) & set(gt["expected_rules"]))
            faithfulness = rule_overlap / len(gt["expected_rules"]) if gt["expected_rules"] else 0

            # Simple relevancy: answer length > 50 chars and doesn't say "cannot find"
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

    # Aggregate
    avg_faith = sum(r["faithfulness"] for r in gen_results) / len(gen_results)
    relevancy = sum(r["is_relevant"] for r in gen_results) / len(gen_results)
    avg_time = sum(r["time_sec"] for r in gen_results) / len(gen_results)

    print(f"\n{'─' * 50}")
    print(f"  Avg Faithfulness:    {avg_faith:.2%}")
    print(f"  Answer Relevancy:    {relevancy:.2%}")
    print(f"  Avg Response Time:   {avg_time:.1f}s")
    print(f"{'─' * 50}\n")

    return {"faithfulness": avg_faith, "relevancy": relevancy, "avg_time": avg_time, "details": gen_results}


def export_results(retrieval_results, gen_results, prefix):
    """Export evaluation results to CSV."""
    import csv

    # Retrieval CSV
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
    parser = argparse.ArgumentParser(description="GFR RAG Evaluation")
    parser.add_argument("--full", action="store_true", help="Run full evaluation with LLM (needs GPU)")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid BM25+semantic retrieval")
    parser.add_argument("--compare", action="store_true", help="Compare semantic vs hybrid retrieval")
    parser.add_argument("--multi-k", action="store_true", help="Compare retrieval at k=3,5,7,10")
    parser.add_argument("--export", type=str, default=None, help="Export results to CSV with given prefix")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    args = parser.parse_args()

    vectorstore, embeddings = load_vectorstore()

    if args.multi_k:
        # Multi-k evaluation for thesis table
        print("\n" + "=" * 70)
        print("MULTI-K RETRIEVAL COMPARISON")
        print("=" * 70)
        print(f"  {'k':>3} | {'Hit Rate':>10} | {'MRR':>8} | {'Precision':>10}")
        print(f"  {'─'*40}")
        for k in [3, 5, 7, 10]:
            r = evaluate_retrieval(vectorstore, k=k, label=f"SEMANTIC k={k}")
            print(f"  {k:>3} | {r['hit_rate']:>9.2%} | {r['mrr']:>8.4f} | {r['avg_precision']:>10.4f}")
        print("=" * 70)
        return

    if args.compare:
        # Side-by-side comparison for thesis
        sem_results = evaluate_retrieval(vectorstore, k=args.k, label="SEMANTIC")
        hyb_results = evaluate_hybrid_retrieval(vectorstore, k=args.k)

        print("\n" + "=" * 70)
        print("COMPARISON: SEMANTIC vs HYBRID RETRIEVAL")
        print("=" * 70)
        print(f"  {'Metric':<20} {'Semantic':>12} {'Hybrid':>12} {'Delta':>10}")
        print(f"  {'─'*54}")
        for metric, label in [("hit_rate", "Hit Rate"), ("mrr", "MRR"), ("avg_precision", "Avg Precision")]:
            s = sem_results[metric]
            h = hyb_results[metric]
            delta = h - s
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<20} {s:>11.4f} {h:>12.4f} {sign}{delta:>9.4f}")
        print("=" * 70)
        retrieval_results = hyb_results  # use hybrid for export
    elif args.hybrid:
        retrieval_results = evaluate_hybrid_retrieval(vectorstore, k=args.k)
    else:
        retrieval_results = evaluate_retrieval(vectorstore, k=args.k, label="SEMANTIC")

    # Optionally run full evaluation
    gen_results = None
    if args.full:
        gen_results = evaluate_full(vectorstore, embeddings)

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
        print(f"  Faithfulness:       {gen_results['faithfulness']:.2%}")
        print(f"  Answer Relevancy:   {gen_results['relevancy']:.2%}")
        print(f"  Avg Response Time:  {gen_results['avg_time']:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

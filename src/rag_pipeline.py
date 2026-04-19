"""
GFR RAG Pipeline - CLI Interface
=================================
Command-line version of the RAG pipeline for testing and evaluation.
Mirrors app.py config: Qwen2.5-14B 8-bit quantization, CPU embeddings,
NAS bypass, and backend logging.
"""

import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import torch
import re
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_rag_chain():
    # ── Embeddings (CPU to save GPU VRAM) ──
    print("[1/4] Loading embedding model (bge-large-en-v1.5) on CPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    # ── ChromaDB with NAS bypass ──
    print("[2/4] Connecting to ChromaDB...")
    local_db_dir = "/tmp/gfr_chroma_db"
    nas_db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

    if not os.path.exists(local_db_dir):
        print(f"  Copying DB from NAS to {local_db_dir}...")
        shutil.copytree(nas_db_dir, local_db_dir)

    vectorstore = Chroma(
        persist_directory=local_db_dir,
        embedding_function=embeddings,
        collection_name="gfr_2025"
    )
    doc_count = vectorstore._collection.count()
    print(f"  Collection 'gfr_2025': {doc_count} documents loaded")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ── LLM with 8-bit quantization ──
    print("[3/4] Loading LLM with 8-bit quantization...")
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1500,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # ── Prompt ──
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

    # ── Format docs with logging ──
    def format_docs(docs):
        print("\n" + "=" * 60)
        print("[BACKEND LOG] RETRIEVED CONTEXT FROM VECTOR DB:")
        print("=" * 60)
        formatted = []
        for i, doc in enumerate(docs):
            doc_str = f"Rule {doc.metadata.get('rule_number', 'N/A')} ({doc.metadata.get('title', 'N/A')}):\n{doc.page_content}"
            print(f"\n--- Document {i+1} | Rule {doc.metadata.get('rule_number', 'N/A')} ---")
            print(doc_str[:300] + "..." if len(doc_str) > 300 else doc_str)
            formatted.append(doc_str)
        return "\n\n".join(formatted)

    # ── RAG Chain ──
    print("[4/4] Building RAG chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def parse_response(response):
    """Clean up the response."""
    return "", response.strip()


# --- Pattern-based Intent Classifier ---
GREETING_PATTERNS = re.compile(
    r'^('
    r'hi|hello|hey|helo|hii+|good morning|good evening|good afternoon|good night|'
    r'howdy|greetings|namaste|thanks|thank you|bye|goodbye|see you|'
    r'how are you|how r you|how r u|how are u|how do you do|'
    r'what\'?s up|whats up|wassup|sup|yo|hola|'
    r'nice to meet you|pleased to meet you'
    r')\s*[\?\!\.\,]*\s*$',
    re.IGNORECASE
)

META_PATTERNS = re.compile(
    r'^('
    r'who are you|what are you|what is this|what do you do|'
    r'what can you do|what can you help|what is your (name|purpose)|'
    r'who (made|built|created|developed) you|tell me about yourself|'
    r'are you (a bot|ai|human|real)|help|what is gfr'
    r')\s*[\?\!\.\,]*\s*$',
    re.IGNORECASE
)

def classify_query(query):
    """Returns 'greeting', 'meta', 'off_topic', or 'gfr'."""
    q = query.strip().lower()
    if len(q) < 3:
        return "greeting"
    if GREETING_PATTERNS.match(q):
        return "greeting"
    if META_PATTERNS.match(q):
        return "meta"
    gfr_keywords = [
        'rule', 'gfr', 'procurement', 'tender', 'grant', 'budget', 'expenditure',
        'audit', 'sanction', 'appropriation', 'government', 'ministry', 'department',
        'financial', 'fund', 'account', 'loss', 'disposal', 'inventory', 'imprest',
        'advance', 'security', 'deposit', 'contract', 'bid', 'gem', 'e-procurement',
        'contingent', 'surplus', 'write off', 'delegation', 'parliament', 'constitution',
        'revenue', 'receipt', 'payment', 'officer', 'gazetted', 'public money',
        'pension', 'salary', 'stores', 'works', 'loan', 'subsidy', 'autonomous',
        'money', 'purchase', 'buying', 'sell', 'goods', 'services', 'vendor',
        'supplier', 'quotation', 'estimate', 'sanction', 'voucher', 'bill',
        'inspection', 'verification', 'stock', 'asset', 'property', 'building',
        'rent', 'lease', 'maintenance', 'repair', 'write-off', 'condemnation',
        'utilization', 'certificate', 'ngo', 'body', 'organization', 'society',
        'consolidated fund', 'contingency', 'supplementary', 'demand',
        'public account', 'treasury', 'comptroller', 'cag', 'accounts',
        'expenditure', 'head of account', 'major head', 'minor head',
    ]
    if any(kw in q for kw in gfr_keywords):
        return "gfr"
    # STRICT: anything without GFR keywords is off-topic
    return "off_topic"


def main():
    rag_chain = build_rag_chain()

    print("\n" + "=" * 60)
    print("GFR RAG Pipeline Ready (Qwen2.5-14B 8-bit, CPU embeddings)")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            query = input("\n> Query: ")
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query.strip():
                continue

            query_type = classify_query(query)

            if query_type == "greeting":
                print("\nHello! I'm the GFR Legal Assistant. Ask me anything about the General Financial Rules (GFR) 2025.")
                continue
            elif query_type == "meta":
                print("\nI'm the GFR Legal Assistant, powered by Qwen 2.5 14B running locally. I can answer questions about the General Financial Rules (GFR) 2025 — procurement, budgeting, grants, expenditure, audits, or any specific rule number.")
                continue
            elif query_type == "off_topic":
                print("\nI'm specifically designed to answer questions about the General Financial Rules (GFR) 2025. Please ask me about GFR topics like procurement, budgeting, grants, expenditure, or any specific rule number.")
                continue

            print(f"\n[BACKEND LOG] USER QUERY: {query}")
            print("Searching and generating...\n")

            response = rag_chain.invoke(query)

            print("\n" + "=" * 60)
            print("[BACKEND LOG] RAW LLM RESPONSE:")
            print("=" * 60)
            print(response)

            thinking, answer = parse_response(response)

            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            if thinking:
                print(f"\n[Reasoning]\n{thinking}\n")
            print(answer)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()

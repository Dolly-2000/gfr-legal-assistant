import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)

import streamlit as st
import torch
import shutil
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure Streamlit page
st.set_page_config(
    page_title="GFR Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ GFR Legal Assistant (Qwen 2.5 14B - 4bit)")
st.markdown("Ask financial and legal questions about the General Financial Rules (GFR) 2025.")


# --------------------------------------------------------------------------------
# Model & DB Loading 
# Important: @st.cache_resource ensures the 54GB model is only loaded ONCE
# and kept in the GPU VRAM across multiple messages and browser refreshes.
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Models into VRAM... (This takes a few minutes on first run)")
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"} # Force embeddings onto CPU RAM to save GPU VRAM
    )
    # --------------------------------------------------------------------------------
    # Bypass NAS Database Locks
    # embed_and_store.py writes to /tmp/gfr_chroma_db (fast local storage).
    # If it doesn't exist yet, copy from the NAS backup.
    # --------------------------------------------------------------------------------
    local_db_dir = "/tmp/gfr_chroma_db"
    nas_db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
    
    if not os.path.exists(local_db_dir):
        print(f"Copying database from NAS to local storage '{local_db_dir}'...")
        shutil.copytree(nas_db_dir, local_db_dir)
        
    vectorstore = Chroma(persist_directory=local_db_dir, embedding_function=embeddings, collection_name="gfr_2025")
    # MMR (Maximum Marginal Relevance): fetches diverse chunks, k=8 ensures
    # multi-part rules (like Rule 2 definitions) get multiple chunks retrieved.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
    )

    model_id = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure 4-bit quantization (faster + less VRAM ~8GB, slight quality tradeoff)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
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
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False
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
    
    # Store retrieved docs for source citations (shared via closure)
    retrieved_docs_store = []

    def format_docs(docs):
        print("\n" + "="*50)
        print("[BACKEND LOG] RETRIEVED CONTEXT FROM VECTOR DB:")
        print("="*50)
        retrieved_docs_store.clear()
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_str = f"Rule {doc.metadata.get('rule_number', 'N/A')} ({doc.metadata.get('title', 'N/A')}):\n{doc.page_content}"
            print(f"\n--- Document {i+1} Target Rule: {doc.metadata.get('rule_number', 'N/A')} ---")
            print(doc_str)
            formatted_docs.append(doc_str)
            retrieved_docs_store.append(doc)
            
        final_context = "\n\n".join(formatted_docs)
        print("\n[BACKEND LOG] SENDING TO LLM AND WAITING FOR GENERATION...\n")
        return final_context

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retrieved_docs_store

# Initialize system
rag_chain, retrieved_docs_store = load_rag_pipeline()

# --------------------------------------------------------------------------------
# Query Classifier — Fast regex-based (NO LLM call, zero overhead)
# Logic: greetings/meta → instant response, no RAG
#        if query has NO GFR-related words → off_topic (default STRICT)
#        otherwise → send to RAG pipeline
# --------------------------------------------------------------------------------
import re as _re

GREETING_PATTERNS = _re.compile(
    r'^('
    r'hi|hello|hey|helo|hii+|good morning|good evening|good afternoon|good night|'
    r'howdy|greetings|namaste|thanks|thank you|bye|goodbye|see you|'
    r'how are you|how r you|how r u|how are u|how do you do|'
    r'what\'?s up|whats up|wassup|sup|yo|hola|'
    r'nice to meet you|pleased to meet you'
    r')\s*[\?\!\.\,]*\s*$',
    _re.IGNORECASE
)

META_PATTERNS = _re.compile(
    r'^('
    r'who are you|what are you|what is this|what do you do|'
    r'what can you (do|help|tell)|what is your (name|purpose)|'
    r'who (made|built|created|developed) you|tell me about yourself|'
    r'are you (a bot|ai|human|real)|help me|what is gfr'
    r')\s*[\?\!\.\,]*\s*$',
    _re.IGNORECASE
)

GREETING_RESPONSE = "Hello! 👋 I'm the GFR Legal Assistant. I can answer questions about the **General Financial Rules (GFR) 2025** — covering procurement, budgeting, expenditure, grants, losses, and more. How can I help you?"

META_RESPONSE = ("I'm the **GFR Legal Assistant**, powered by **Qwen 2.5 14B** running locally. "
    "I can answer questions about the **General Financial Rules (GFR) 2025** of India — "
    "including topics like:\n\n"
    "- 📋 **Procurement & Tenders** (GeM, e-procurement, bid security)\n"
    "- 💰 **Budget & Expenditure** (appropriation, sanctions, re-appropriation)\n"
    "- 🏛️ **Grants-in-Aid** (conditions, utilization certificates)\n"
    "- 📦 **Inventory & Stores** (disposal, write-off, physical verification)\n"
    "- 📑 **Any specific Rule number** (e.g., 'What does Rule 149 say?')\n\n"
    "Just type your question below!")

def classify_query(query):
    """Returns 'greeting', 'meta', or 'rag'. Everything non-greeting goes to RAG."""
    q = query.strip()
    if len(q) < 3 or GREETING_PATTERNS.match(q.lower()):
        return "greeting"
    if META_PATTERNS.match(q.lower()):
        return "meta"
    return "rag"

# --------------------------------------------------------------------------------
# Chat UI Logic
# --------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Source Rules Retrieved"):
                for src in msg["sources"]:
                    st.markdown(f"**Rule {src['rule']}** — {src['title'][:80]}")

# Sidebar with system info
with st.sidebar:
    st.header("System Info")
    st.markdown("""
    - **LLM**: Qwen2.5-14B-Instruct (4-bit NF4)
    - **Embeddings**: BGE-Large-EN-v1.5
    - **Vector DB**: ChromaDB (565 docs)
    - **Retrieval**: Semantic MMR (k=8)
    """)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# User Input
if prompt_text := st.chat_input("Ask a question about the GFR..."):
    print(f"\n\n\n{'='*70}\n[BACKEND LOG] NEW USER QUERY: {prompt_text}\n{'='*70}")
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Generate assistant response
    with st.chat_message("assistant"):
        query_type = classify_query(prompt_text)
        print(f"[BACKEND LOG] Query classified as: {query_type}")
        
        if query_type == "greeting":
            final_answer = GREETING_RESPONSE
            sources = []
            st.markdown(final_answer)
        elif query_type == "meta":
            final_answer = META_RESPONSE
            sources = []
            st.markdown(final_answer)
        else:
            with st.spinner("Analyzing rules and generating response..."):
                response = rag_chain.invoke(prompt_text)
                
                print("\n" + "="*50)
                print("[BACKEND LOG] RAW LLM GENERATED RESPONSE:")
                print("="*50)
                print(response)
                print("="*50 + "\n")
                
                final_answer = response.strip()
                st.markdown(final_answer)
                
                # Source citations
                sources = []
                seen_rules = set()
                for doc in retrieved_docs_store:
                    rule_num = doc.metadata.get("rule_number", "N/A")
                    if rule_num not in seen_rules:
                        seen_rules.add(rule_num)
                        sources.append({
                            "rule": rule_num,
                            "title": doc.metadata.get("title", "N/A"),
                        })
                
                if sources:
                    with st.expander("📚 Source Rules Retrieved"):
                        for src in sources:
                            st.markdown(f"**Rule {src['rule']}** — {src['title'][:80]}")
            
    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_answer,
        "sources": sources
    })

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

st.title("⚖️ GFR Legal Assistant (Qwen 2.5 14B)")
st.markdown("Ask financial and legal questions about the General Financial Rules.")

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    model_id = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure 8-bit quantization (better quality than 4-bit, ~16GB VRAM)
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
    return rag_chain, retrieved_docs_store, pipe

# Initialize system
rag_chain, retrieved_docs_store, raw_pipe = load_rag_pipeline()

# --------------------------------------------------------------------------------
# Query Classifier — LLM-based intent classification
# Only simple greetings use regex (for instant response). Everything else
# goes through the LLM to decide if it's GFR-related or off-topic.
# --------------------------------------------------------------------------------
import re as _re

# Regex ONLY for obvious greetings (instant, no LLM needed)
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

OFF_TOPIC_RESPONSE = ("I'm specifically designed to answer questions about the **General Financial Rules (GFR) 2025**. "
    "I can't help with that topic, but feel free to ask me about GFR rules like procurement, "
    "budgeting, grants, expenditure, audits, or any specific rule number.")

def classify_query(query):
    """Uses the LLM to classify query intent. Returns 'greeting', 'meta', 'off_topic', or 'gfr'."""
    q = query.strip()
    # Instant: obvious greetings (no need to waste LLM time)
    if len(q) < 3 or GREETING_PATTERNS.match(q.lower()):
        return "greeting"

    # Use LLM for everything else — same model already in VRAM, just generate ~5 tokens
    classification_prompt = f"""<|im_start|>system
You are a query classifier for a GFR (General Financial Rules) legal assistant chatbot.
Classify the user's query into exactly one category. Reply with ONLY the category name, nothing else.

Categories:
- gfr_question: Any question related to government financial rules, procurement, budgeting, expenditure, tenders, grants, audits, sanctions, accounts, government finance, or Indian financial regulations.
- meta_question: Questions about the chatbot itself (who are you, what can you do, help, etc.)
- off_topic: Anything else — general knowledge, personal questions, weather, coding, sports, etc.
<|im_end|>
<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant
"""
    result = raw_pipe(classification_prompt, max_new_tokens=10, temperature=0.01)[0]["generated_text"].strip().lower()
    print(f"[BACKEND LOG] LLM classifier raw output: '{result}'")

    if "gfr" in result:
        return "gfr"
    elif "meta" in result:
        return "meta"
    else:
        return "off_topic"

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
    - **LLM**: Qwen2.5-14B-Instruct (8-bit)
    - **Embeddings**: BGE-Large-EN-v1.5
    - **Vector DB**: ChromaDB (565 docs)
    - **Retrieval**: Semantic Search (k=5)
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
        elif query_type == "off_topic":
            final_answer = OFF_TOPIC_RESPONSE
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

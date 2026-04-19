# GFR Legal Assistant — Complete Setup Guide
## RAG-based Question Answering System for General Financial Rules (GFR) 2025

---

## 📋 Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** system that answers questions about India's General Financial Rules (GFR) 2025. It parses the GFR PDF, chunks the rules, embeds them into a vector database, and uses a Large Language Model (LLM) to generate answers with source citations.

**Tech Stack:**
- **LLM**: Qwen2.5-14B-Instruct (4-bit quantized) / Qwen2.5-7B-Instruct (4-bit)
- **Embeddings**: BAAI/bge-large-en-v1.5
- **Vector DB**: ChromaDB
- **Framework**: LangChain + HuggingFace Transformers
- **UI**: Streamlit
- **GPU**: NVIDIA H100 NVL (80GB) or any GPU with ≥8GB VRAM

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB (7B 4-bit) | 16 GB (14B 4-bit) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |
| CUDA | 11.8+ | 12.1+ |

**Our setup:** 2x NVIDIA H100 NVL (95 GB each) on shared server `aigpu (10.71.9.36)`

---

## 🚀 Step-by-Step Setup (Fresh Machine)

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git

# Verify CUDA (must already have NVIDIA drivers + CUDA toolkit)
nvidia-smi
# Should show your GPU(s) and CUDA version ≥ 11.8
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/Dolly-2000/gfr-legal-assistant.git
cd gfr-legal-assistant
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install PyTorch (GPU version)

```bash
# For CUDA 12.1+ (check with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Install Project Dependencies

```bash
pip install -r requirements.txt

# Additional required packages (may not be in requirements.txt yet)
pip install bitsandbytes streamlit transformers
```

### Step 6: Download the GFR PDF

Place the GFR 2025 PDF in:
```
data/raw_pdfs/GFR2025.pdf
```

---

## 📂 Project File Structure

```
Thesis/
├── data/
│   ├── raw_pdfs/          # Original GFR 2025 PDF
│   │   └── GFR2025.pdf
│   ├── parsed/            # Chunked JSON (generated)
│   │   └── 2025_GFR_chunks.json
│   └── chroma_db/         # Vector database (generated)
│       └── gfr_2025/
├── src/
│   ├── chunk_gfr_v2.py    # Step 1: Parse PDF → JSON chunks
│   ├── embed_and_store.py # Step 2: Embed chunks → ChromaDB
│   ├── app_4bit.py        # Step 3: Main app (14B, 4-bit) ← USE THIS
│   ├── app_7b.py          # Alt: Smaller model (7B, 4-bit)
│   ├── app_v2.py          # Alt: 8-bit version (larger VRAM)
│   ├── app.py             # Original (deprecated, slow)
│   ├── evaluate.py        # Evaluation framework (8-bit)
│   ├── evaluate_4bit.py   # Evaluation framework (4-bit)
│   ├── rag_pipeline.py    # CLI RAG pipeline
│   └── hybrid_retriever.py # BM25 + semantic hybrid
├── notebooks/             # Jupyter experiments
├── requirements.txt       # Python dependencies
├── .gitignore
└── SETUP_GUIDE.md         # This file
```

---

## 🔄 Pipeline Execution Order

### Run Once (Data Preparation):

```bash
# Step 1: Parse GFR PDF into rule chunks
python3 src/chunk_gfr_v2.py
# Output: data/parsed/2025_GFR_chunks.json (324 rules, 565 chunks)

# Step 2: Embed chunks into ChromaDB
python3 src/embed_and_store.py
# Output: /tmp/gfr_chroma_db/ (565 documents embedded)
# Also copies to: data/chroma_db/ (NAS backup)
```

### Run Every Time (Application):

```bash
# Set GPU (check nvidia-smi for free GPU)
export CUDA_VISIBLE_DEVICES=0

# Run the main app (14B 4-bit — recommended)
streamlit run src/app_4bit.py --server.port 8501 --server.address 0.0.0.0 \
    --server.fileWatcherType none --logger.level error

# OR: Run 7B model (faster, less accurate)
streamlit run src/app_7b.py --server.port 8502 --server.address 0.0.0.0 \
    --server.fileWatcherType none --logger.level error
```

Access the UI at: `http://<server-ip>:8501`

---

## 📊 Running Evaluations

```bash
# Retrieval-only evaluation (no GPU needed, fast)
python3 src/evaluate.py

# Full evaluation with 4-bit 14B model
python3 src/evaluate_4bit.py --full --export results_4bit_14b

# Full evaluation with 4-bit 7B model  
python3 src/evaluate_4bit.py --full --model 7b --export results_4bit_7b

# Compare retrieval at different k values
python3 src/evaluate.py --multi-k

# Compare semantic vs hybrid retrieval
python3 src/evaluate.py --compare
```

---

## 🧠 Model Configurations Explained

| Config | Model | Quant | VRAM | Speed | Quality |
|--------|-------|-------|------|-------|---------|
| `app_4bit.py` | 14B | 4-bit NF4 | ~8 GB | ~2 min | ★★★★☆ |
| `app_7b.py` | 7B | 4-bit NF4 | ~4 GB | ~30s | ★★★☆☆ |
| `app_v2.py` | 14B | 8-bit | ~16 GB | ~3 min | ★★★★★ |

**Recommendation:** Use `app_4bit.py` for demos. Use `app_v2.py` if GPU has 20+ GB free.

---

## 🔧 Key Technical Decisions

### 1. Retrieval Strategy: MMR (Maximum Marginal Relevance)
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
)
```
- `k=8`: Retrieve 8 diverse chunks (covers multi-part rules like Rule 2 definitions)
- `fetch_k=20`: Consider 20 candidates before diversity filtering
- `lambda_mult=0.7`: Balance relevance (1.0) vs diversity (0.0)

### 2. Quantization: 4-bit NF4
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,    # Second quantization for further savings
)
```

### 3. Prompt Engineering
The system prompt explicitly instructs the LLM to:
- Extract ALL definitions (not just the first one found)
- Combine information from multiple chunks
- Cite specific rule numbers
- Use structured formatting (bullets/numbered lists)

### 4. Classifier: Minimal (greeting-only)
Only greetings and meta-questions are intercepted. Everything else goes to RAG. This avoids false-positive filtering that was blocking legitimate GFR queries.

---

## ⚠️ Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| 30+ min per answer | GPU memory full, model swapping to CPU | Check `nvidia-smi`, use a free GPU with `CUDA_VISIBLE_DEVICES` |
| "CUDA out of memory" | Not enough VRAM | Switch to 4-bit model or kill other GPU processes |
| ChromaDB not found | First run, DB not copied | Run `python3 src/embed_and_store.py` first |
| torchvision warnings | Harmless import warnings | Use `--logger.level error` flag |
| Definitions incomplete | Old prompt was too vague | Already fixed in latest version |
| Query classified as off-topic | Old regex classifier too strict | Already removed in latest version |

---

## 📈 Evaluation Results

| Metric | Value (8-bit) |
|--------|---------------|
| Questions | 30 |
| Hit Rate@5 | 100.00% |
| MRR@5 | 0.8178 |
| Faithfulness | 72.78% |
| Answer Relevancy | 100.00% |

*(Run `evaluate_4bit.py --full` to get 4-bit results)*

---

## 🌐 Remote Access (Our Server Setup)

```bash
# SSH to the GPU server
ssh dolly@10.71.9.36

# Use tmux to keep processes running after disconnect
tmux new -s thesis
# OR attach to existing session:
tmux attach -t thesis

# Check GPU availability
nvidia-smi

# Run the app
export CUDA_VISIBLE_DEVICES=0
streamlit run src/app_4bit.py --server.port 8501 --server.address 0.0.0.0 \
    --server.fileWatcherType none --logger.level error

# Detach from tmux (keeps running): Ctrl+B then D
```

---

## 📦 Pushing to GitHub

```bash
cd ~/Thesis

# Initialize git (first time only)
git init
git branch -M main

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/gfr-legal-assistant.git

# Check what will be tracked (should NOT include venv/ or chroma_db/)
git status

# Add and commit
git add .
git commit -m "GFR Legal Assistant - RAG pipeline with Qwen2.5"

# Push
git push -u origin main
```

**Important:** The `.gitignore` excludes `venv/`, `data/chroma_db/`, `data/parsed/`, and `__pycache__/`. The raw PDF and source code will be tracked.

---

## 🎬 Video Recording Checklist

For your setup video, cover these steps in order:

1. **Show hardware** — `nvidia-smi` output, GPU specs
2. **Clone repo** — `git clone ...`
3. **Create venv** — `python3 -m venv venv && source venv/bin/activate`
4. **Install torch** — `pip install torch --index-url ...`
5. **Install deps** — `pip install -r requirements.txt`
6. **Show PDF** — `ls data/raw_pdfs/`
7. **Parse PDF** — `python3 src/chunk_gfr_v2.py` (show output: 324 rules, 565 chunks)
8. **Embed to DB** — `python3 src/embed_and_store.py` (show output: 565 docs stored)
9. **Run app** — `streamlit run src/app_4bit.py ...`
10. **Demo queries** — Ask 3-4 questions, show source citations
11. **Run eval** — `python3 src/evaluate_4bit.py --full`
12. **Show results** — Hit Rate, MRR, Faithfulness scores

Total recording time: ~15-20 minutes

---

## 📁 File Descriptions

| File | Purpose | When to Run |
|------|---------|-------------|
| `chunk_gfr_v2.py` | Parses GFR PDF into 565 JSON chunks using two-zone detection | Once (data prep) |
| `embed_and_store.py` | Embeds chunks using BGE-Large and stores in ChromaDB | Once (data prep) |
| `app_4bit.py` | **Main Streamlit app** — 14B model, 4-bit, pink UI | Every time (demo) |
| `app_7b.py` | Smaller 7B model variant for comparison | For comparison |
| `app_v2.py` | 8-bit variant (better quality, needs more VRAM) | If GPU has 20GB+ free |
| `evaluate.py` | Evaluation with 30 ground-truth questions (8-bit) | For thesis metrics |
| `evaluate_4bit.py` | Evaluation for 4-bit models (14B and 7B) | For thesis metrics |
| `hybrid_retriever.py` | BM25 + semantic hybrid retrieval | Optional experiment |
| `rag_pipeline.py` | CLI interface for RAG (no Streamlit) | Testing/debugging |

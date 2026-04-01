# ⚖️ LawGPT-IN: Fine-Tuned Indian Legal LLM 

![Status](https://img.shields.io/badge/Status-Active-success)
![LLM](https://img.shields.io/badge/LLM-Mistral--7B-blue)
![FineTuning](https://img.shields.io/badge/FineTuning-LoRA%20(Unsloth)-orange)

**LawGPT-IN** is a specialized, fine-tuned Large Language Model (Mistral-7B) designed specifically for the complexities of the Indian judicial system. It predicts case outcomes and retrieves highly relevant legal precedents using a custom Hybrid RAG architecture.

## 🌟 The Problem
Indian legal language is highly unique—it blends standard British English, archaic legal Latin, and directly translated regional phrases. Standard LLMs (like pure ChatGPT or open-source models) struggle heavily with generating accurate citations for the **Indian Penal Code (IPC)**, **CrPC**, and interpreting the **Constitution of India**.

India currently has over **45 million pending cases**. A highly contextual, functional AI demo in this space serves as a powerful foundational tool for LegalTech disruption (e.g., startups like SpotDraft, Leegality, and NyaySetu).

## 🛠️ Core Tech Stack
* **Base Model:** Mistral-7B-Instruct-v0.2
* **Fine-Tuning:** LoRA Adapters over 4-bit quantization, optimized using [Unsloth](https://github.com/unslothai/unsloth) for extreme memory efficiency on a single free T4 GPU.
* **Dataset:** 500K+ scraped judgments from the Indian Kanoon corpus.
* **Retrieval/RAG:** Hybrid Retrieval Layer
  * **Sparse (BM25):** For exact statute & penal code matching.
  * **Dense (FAISS / Weaviate):** For semantic, context-aware precedent retrieval. 
* **Backend:** LangChain
* **Frontend:** Next.js Dashboard

---

## 🚀 Project Pipeline

This project is divided into three major stages:

### Stage 1: The Dataset Pipeline `(Folder: dataset/)`
Unlike generalized models, this model thrives on clean, structured Indian legal instruction pairs. 
1. **Scraping:** Automated retrieval engine querying the Indian Kanoon API for case types (Criminal Appeals, Bail Applications, Motor Accident Claims, etc.).
2. **Cleaning:** Heuristic NLP chunking splits unstructured judgements into `Facts`, `Issues`, `Reasoning`, and `Verdict`.
3. **ChatML Formatting:** Generating rich synthetic training pairs (Summary Generation, Verdict Prediction, and Acts Cited), compiled into a clean JSONL dataset ready for HuggingFace.

### Stage 2: LoRA Fine-Tuning `(Folder: finetuning/)`
Using a standard Colab T4 GPU, the 7-Billion parameter Mistral model is actively fine-tuned.
1. **Unsloth Optimization:** Fast loading with 4-bit `bitsandbytes` quantization.
2. **Parameter Injection:** LoRA adapters injected across all linear layers (`q_proj`, `v_proj`, etc.) with Rank=16.
3. **Training:** `SFTTrainer` consumes the ChatML dataset, teaching Mistral not just how to speak, but how to analyze facts and output judgments mimicking Indian supreme/high courts. 
4. **Adapter Push:** The newly mapped 100MB weights are seamlessly pushed back to the HuggingFace Hub.

### Stage 3: Hybrid RAG Architecture & UI *(In Progress)*
Fine-tuning teaches the model *how* to speak like an Indian lawyer, but RAG gives it the memory of the entire constitution exactly as it stands today.
1. **Dual Retrieval VectorDB:** Combining BM25 keyword matching for specific statutes with Weaviate Dense Vector embeddings to fetch highly similar case backgrounds.
2. **Next.js Dashboard:** A sleek, minimal web interface where users input case "Facts", and get back legal reasoning, predicted verdicts, and cited precedents.

---

## 💻 How to Replicate / Run

### 1. Data Pipeline Setup
* Clone the repository and install dependencies: `pip install -r requirements.txt`
* Insert your Indian Kanoon API Token and HuggingFace Write Token in `config.py`
* Run the scraper and cleaning scripts to push your localized dataset straight to your HuggingFace account.

### 2. Fine-Tuning on Colab
* We've included the complete unsloth finetuning notebook in `notebooks/LawGPT_IN_LoRA_Finetune.ipynb`.
* Upload to Google Colab, select the **T4 Free GPU**.
* *(If you face a HuggingFace timeout in Colab, ensure you enable the `modelscope` parameter fallback built into our script!)*
* Once finished, the adapters automatically sync to HuggingFace.

---

## 🔮 Future Roadmap
- Expand the dataset to include regional tribunal rulings (NCLT, ITAT).
- Host the complete Web App using HuggingFace Spaces / Vercel.
- Implement streaming tokens for the frontend UI.

> *Disclaimer: This project is an academic AI research tool and should not be used as a replacement for certified legal counsel.*

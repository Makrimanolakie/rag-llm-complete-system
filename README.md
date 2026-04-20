# RAG-LLM Complete System (Timeout Fixed) 🚀

**Πλήρες σύστημα Retrieval-Augmented Generation (RAG) με τοπικό ή cloud LLM για ερωτήσεις σε PDFs.**

Ελληνική & Αγγλική υποστήριξη | Αυτόματη εγκατάσταση | Ollama + Anthropic + OpenAI

## ✨ Χαρακτηριστικά

- **Αυτόματη εγκατάσταση** όλων των dependencies
- **Δύο εκδόσεις**: μία με **fixed timeout 300 δευτερόλεπτα** για Ollama (συνιστάται)
- **Semantic search** με SentenceTransformers
- **Πλήρης RAG pipeline** (Embedder → Indexer → Retriever → LLM Generator)
- Υποστήριξη **Ollama (τοπικό)**, Anthropic Claude, OpenAI GPT-4 ή Template mode
- Αποθήκευση/φόρτωση index για γρήγορη επαναχρησιμοποίηση
- Διαδραστικό CLI menu
- Δείγματα PDFs (Transformer paper + Manifold Destiny άρθρο)

## 🚀 Γρήγορη Εγκατάσταση

```bash
git clone https://github.com/USERNAME/rag-llm-complete-system.git
cd rag-llm-complete-system
python complete_rag_llm_implementation_timeout_fixed.py

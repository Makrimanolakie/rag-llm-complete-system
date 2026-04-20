"""
COMPLETE RAG-LLM SYSTEM - TIMEOUT FIXED
========================================
Fixed: Ollama timeout increased to 300 seconds (5 minutes)
First query loads model into memory - takes time but works!
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def install_dependencies():
    """Check and install required dependencies."""
    required = {
        'fitz': 'PyMuPDF',
        'sentence_transformers': 'sentence-transformers',
        'numpy': 'numpy',
        'requests': 'requests'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 Installing: {', '.join(missing)}")
        import subprocess
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print("✓ Installation complete!\n")

install_dependencies()

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Embedder loaded (dimension: {self.dimension})")
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=show_progress, 
                                convert_to_numpy=True, normalize_embeddings=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text], show_progress=False)[0]

class DocumentIndexer:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = None
    
    def extract_pdf_text(self, pdf_path: Path) -> Dict[int, str]:
        page_texts = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page_texts[page_num] = doc[page_num].get_text()
            doc.close()
        except Exception as e:
            print(f"✗ Error reading {pdf_path}: {e}")
        return page_texts
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        return chunks
    
    def index_directory(self, pdf_directory: str) -> bool:
        print(f"\n{'='*70}")
        print(f"📚 INDEXING DOCUMENTS")
        print(f"{'='*70}\n")
        
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            print(f"✗ Directory not found: {pdf_directory}")
            return False
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"✗ No PDF files found")
            return False
        
        print(f"Found {len(pdf_files)} PDF(s)\n")
        
        self.chunks = []
        self.chunk_metadata = []
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"[{idx}/{len(pdf_files)}] {pdf_path.name}")
            page_texts = self.extract_pdf_text(pdf_path)
            
            for page_num, page_text in page_texts.items():
                if len(page_text.strip()) < 50:
                    continue
                page_chunks = self.chunk_text(page_text)
                for chunk_idx, chunk in enumerate(page_chunks):
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        'pdf_name': pdf_path.name,
                        'page_num': page_num,
                        'chunk_idx': chunk_idx
                    })
            print(f"    ✓ {len(page_texts)} pages")
        
        print(f"\n✓ Created {len(self.chunks)} chunks")
        print("\n🔮 Generating embeddings...")
        self.embeddings = self.embedder.encode(self.chunks)
        print(f"✓ Embeddings: {self.embeddings.shape}")
        print(f"\n{'='*70}")
        print(f"✅ INDEXING COMPLETE")
        print(f"{'='*70}\n")
        return True
    
    def save_index(self, filepath: str):
        data = {
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata,
            'embeddings': self.embeddings.tolist(),
            'saved_at': datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"💾 Saved: {filepath}")
    
    def load_index(self, filepath: str):
        print(f"📂 Loading: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.chunk_metadata = data['chunk_metadata']
        self.embeddings = np.array(data['embeddings'])
        print(f"✓ Loaded {len(self.chunks)} chunks")

class Retriever:
    def __init__(self, indexer: DocumentIndexer, embedder: Embedder):
        self.indexer = indexer
        self.embedder = embedder
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.indexer.embeddings is None:
            raise ValueError("No documents indexed")
        query_embedding = self.embedder.encode_single(query)
        similarities = np.dot(self.indexer.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            idx = int(idx)
            results.append({
                'chunk': self.indexer.chunks[idx],
                'metadata': self.indexer.chunk_metadata[idx],
                'score': float(similarities[idx])
            })
        return results

class LLMGenerator:
    def __init__(self, mode: str = "local"):
        self.mode = mode
        self.client = None
        print(f"🤖 Initializing LLM: {mode.upper()} mode")
        
        if mode == "local":
            self._init_ollama()
        elif mode == "anthropic":
            self._init_anthropic()
        elif mode == "openai":
            self._init_openai()
        elif mode == "template":
            print("✓ Template mode")
    
    def _init_ollama(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    self.ollama_model = models[0]['name']
                    print(f"✓ Ollama connected ({self.ollama_model})")
                else:
                    print("⚠ No models. Run: ollama pull llama2")
                    self.mode = "template"
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"⚠ Ollama unavailable: {e}")
            print("  Using template mode")
            self.mode = "template"
    
    def _init_anthropic(self):
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                print("✓ Anthropic connected")
            else:
                print("⚠ ANTHROPIC_API_KEY not set")
                self.mode = "template"
        except ImportError:
            print("⚠ pip install anthropic")
            self.mode = "template"
    
    def _init_openai(self):
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                print("✓ OpenAI connected")
            else:
                print("⚠ OPENAI_API_KEY not set")
                self.mode = "template"
        except ImportError:
            print("⚠ pip install openai")
            self.mode = "template"
    
    def generate(self, query: str, context: str, max_tokens: int = 1000) -> str:
        prompt = f"""Based on the following context from documents, answer the question clearly and concisely.

Context:
{context[:3000]}

Question: {query}

Provide a clear answer based on the context above. Be specific and cite information."""

        if self.mode == "local":
            return self._generate_ollama(prompt, max_tokens)
        elif self.mode == "anthropic":
            return self._generate_anthropic(prompt, max_tokens)
        elif self.mode == "openai":
            return self._generate_openai(prompt, max_tokens)
        else:
            return self._generate_template(query, context)
    
    def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        """🔧 FIXED: Increased timeout to 300 seconds (5 minutes)"""
        try:
            print("\n    ⏳ Generating with Ollama...")
            print("    💡 First query takes 1-2 minutes (loads model)")
            print("    ⚡ Next queries will be MUCH faster (5-10 sec)")
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3
                    }
                },
                timeout=300  # 🔧 FIXED: 60 → 300 seconds (5 minutes)
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                print("    ✓ Ollama generated answer successfully!")
                return result
            else:
                print(f"    ⚠ Status {response.status_code}")
                return self._generate_template("", prompt)
                
        except requests.exceptions.Timeout:
            print("\n    ❌ Still timed out after 5 minutes")
            print("    💡 Try smaller model: ollama pull phi")
            return self._generate_template("", prompt)
        except Exception as e:
            print(f"\n    ❌ Ollama error: {e}")
            return self._generate_template("", prompt)
    
    def _generate_anthropic(self, prompt: str, max_tokens: int) -> str:
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"⚠ Anthropic error: {e}")
            return self._generate_template("", prompt)
    
    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a technical expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ OpenAI error: {e}")
            return self._generate_template("", prompt)
    
    def _generate_template(self, query: str, context: str) -> str:
        """Fallback: Extract text when LLM fails."""
        sentences = re.split(r'[.!?]+', context)
        relevant = [s.strip() for s in sentences if len(s.strip()) > 30][:5]
        if relevant:
            return "Based on the documents:\n\n" + "\n".join(f"• {s}." for s in relevant)
        else:
            return "Information found in context. Review sources."

class RAGPipeline:
    def __init__(self, mode: str = "local"):
        print(f"\n{'='*70}")
        print("🚀 INITIALIZING RAG PIPELINE")
        print(f"{'='*70}\n")
        
        self.embedder = Embedder()
        self.indexer = DocumentIndexer(self.embedder)
        self.retriever = Retriever(self.indexer, self.embedder)
        self.generator = LLMGenerator(mode)
        
        print(f"\n✅ RAG Pipeline ready ({mode.upper()} mode)")
        print(f"{'='*70}\n")
    
    def index_documents(self, pdf_directory: str) -> bool:
        return self.indexer.index_directory(pdf_directory)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        print(f"\n{'='*70}")
        print(f"💬 PROCESSING QUERY")
        print(f"{'='*70}\n")
        print(f"Question: {question}\n")
        
        print("🔍 Retrieving relevant content...")
        results = self.retriever.search(question, top_k=top_k)
        print(f"✓ Found {len(results)} relevant chunks\n")
        
        if not results:
            return {
                'answer': "No relevant information found.",
                'sources': [],
                'query': question
            }
        
        print("📝 Preparing context...")
        context_parts = []
        for idx, item in enumerate(results, 1):
            metadata = item['metadata']
            chunk = item['chunk']
            score = item['score']
            context_parts.append(
                f"[Source {idx}] (Relevance: {score:.2f})\n"
                f"Document: {metadata['pdf_name']}, Page {metadata['page_num'] + 1}\n"
                f"{chunk}\n"
            )
        context = "\n---\n".join(context_parts)
        
        print(f"🤖 Generating answer ({self.generator.mode.upper()} mode)...")
        answer = self.generator.generate(question, context)
        
        sources = []
        seen = set()
        for item in results:
            metadata = item['metadata']
            key = (metadata['pdf_name'], metadata['page_num'])
            if key not in seen:
                sources.append({
                    'document': metadata['pdf_name'],
                    'page': metadata['page_num'] + 1,
                    'relevance': item['score']
                })
                seen.add(key)
        
        print(f"\n✅ Answer generated!")
        print(f"{'='*70}\n")
        
        return {'answer': answer, 'sources': sources, 'query': question}
    
    def save_index(self, filepath: str):
        self.indexer.save_index(filepath)
    
    def load_index(self, filepath: str):
        self.indexer.load_index(filepath)

def print_menu():
    print("\n" + "="*70)
    print("🎯 RAG-LLM SYSTEM - MAIN MENU")
    print("="*70)
    print("\n🔧 SETUP:")
    print("  1. Choose LLM mode")
    print("  2. Index PDF documents")
    print("  3. Load saved index")
    print("\n💬 QUERY:")
    print("  4. Ask a question (RAG)")
    print("  5. Search documents only")
    print("\n💾 MANAGEMENT:")
    print("  6. Save index")
    print("  7. View statistics")
    print("\n  0. Exit")
    print("="*70)

def main():
    print("\n" + "🎉"*25)
    print("COMPLETE RAG-LLM SYSTEM - TIMEOUT FIXED")
    print("🎉"*25)
    
    rag = None
    
    while True:
        print_menu()
        choice = input("\n👉 Enter choice: ").strip()
        
        if choice == '1':
            print("\n🔧 SELECT LLM MODE:")
            print("  1. Local (Ollama - Private, free)")
            print("  2. Anthropic Claude (Best quality)")
            print("  3. OpenAI GPT-4 (Good quality)")
            print("  4. Template (No LLM, instant)")
            
            mode_choice = input("\nChoice (1-4): ").strip()
            mode_map = {'1': 'local', '2': 'anthropic', '3': 'openai', '4': 'template'}
            mode = mode_map.get(mode_choice, 'template')
            rag = RAGPipeline(mode=mode)
        
        elif choice == '2':
            if not rag:
                print("\n⚠️ Choose mode first (option 1)")
                continue
            pdf_dir = input("\n📁 PDF directory: ").strip()
            if pdf_dir:
                rag.index_documents(pdf_dir)
                input("\nPress Enter...")
        
        elif choice == '3':
            if not rag:
                rag = RAGPipeline(mode='template')
            filepath = input("\n📂 Index file: ").strip()
            if filepath and Path(filepath).exists():
                rag.load_index(filepath)
                input("\nPress Enter...")
            else:
                print(f"\n✗ Not found: {filepath}")
        
        elif choice == '4':
            if not rag:
                print("\n⚠️ Initialize first (option 1)")
                continue
            if not rag.indexer.chunks:
                print("\n⚠️ Index documents first (option 2)")
                continue
            
            question = input("\n💬 Your question: ").strip()
            if question:
                result = rag.query(question)
                print(f"\n{'='*70}")
                print("ANSWER:")
                print("="*70)
                print(f"\n{result['answer']}\n")
                print("="*70)
                print("📚 SOURCES:")
                for source in result['sources']:
                    print(f"  • {source['document']} (Page {source['page']}) "
                          f"[Relevance: {source['relevance']:.2f}]")
                print("="*70)
                input("\nPress Enter...")
        
        elif choice == '5':
            if not rag or not rag.indexer.chunks:
                print("\n⚠️ Index documents first")
                continue
            query = input("\n🔍 Search: ").strip()
            if query:
                results = rag.retriever.search(query)
                print(f"\n{'='*70}")
                print("SEARCH RESULTS")
                print("="*70)
                for idx, item in enumerate(results, 1):
                    metadata = item['metadata']
                    print(f"\n[{idx}] {metadata['pdf_name']} (Page {metadata['page_num'] + 1})")
                    print(f"    Relevance: {item['score']:.2f}")
                    print(f"    {item['chunk'][:200]}...")
                    print("-"*70)
                input("\nPress Enter...")
        
        elif choice == '6':
            if not rag or not rag.indexer.chunks:
                print("\n⚠️ Nothing to save")
                continue
            filename = input("\n💾 Filename: ").strip() or "rag_index.json"
            rag.save_index(filename)
            input("\nPress Enter...")
        
        elif choice == '7':
            if not rag:
                print("\n⚠️ No system initialized")
                continue
            print(f"\n{'='*70}")
            print("📊 STATISTICS")
            print("="*70)
            print(f"  LLM Mode: {rag.generator.mode.upper()}")
            print(f"  Chunks: {len(rag.indexer.chunks)}")
            print(f"  Embedding dim: {rag.embedder.dimension}")
            if rag.indexer.chunk_metadata:
                docs = set(m['pdf_name'] for m in rag.indexer.chunk_metadata)
                print(f"  Documents: {len(docs)}")
                for doc in sorted(docs):
                    count = sum(1 for m in rag.indexer.chunk_metadata if m['pdf_name'] == doc)
                    print(f"    - {doc}: {count} chunks")
            print("="*70)
            input("\nPress Enter...")
        
        elif choice == '0':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("\n❌ Invalid choice")

if __name__ == "__main__":
    main()

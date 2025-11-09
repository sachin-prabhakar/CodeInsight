"""Hybrid retrieval with BM25 + dense embeddings."""
import os
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter
import math

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from rapidfuzz import fuzz
try:
    from llm import get_embed_provider
    from chunking import CodeChunk, DocChunk
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from llm import get_embed_provider
    from chunking import CodeChunk, DocChunk


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source_type: str  # "code" or "docs"


class HybridRetriever:
    """Hybrid BM25 + dense retrieval."""
    
    def __init__(self, index_dir: str, config: Dict[str, Any]):
        self.index_dir = index_dir
        self.config = config
        self.bm25_weight = config.get("bm25_weight", 0.3)
        self.dense_weight = config.get("dense_weight", 0.7)
        self.rerank = config.get("rerank", "auto")
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed")
        
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(index_dir, "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.code_collection = self.chroma_client.get_collection("code_index")
        except:
            self.code_collection = None
        
        try:
            self.docs_collection = self.chroma_client.get_collection("docs_index")
        except:
            self.docs_collection = None
        
        self.embed_provider = get_embed_provider(
            provider=config.get("provider", "auto"),
            embed_model=config.get("embed_model")
        )
        
        self.cross_encoder = None
        if self.rerank == "auto" and CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except:
                pass
    
    def _bm25_score(self, query_terms: List[str], doc_terms: List[str], doc_freq: Dict[str, int], total_docs: int) -> float:
        """Calculate BM25 score."""
        k1 = 1.5
        b = 0.75
        avg_doc_length = 100
        
        score = 0.0
        doc_term_freq = Counter(doc_terms)
        doc_length = len(doc_terms)
        
        for term in query_terms:
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                df = doc_freq.get(term, 1)
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                numerator = idf * tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += numerator / denominator
        
        return score
    
    def _simple_bm25(self, query: str, documents: List[str]) -> List[float]:
        """Simple BM25 scoring using rapidfuzz as fallback."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        scores = []
        for doc in documents:
            doc_lower = doc.lower()
            score = fuzz.partial_ratio(query_lower, doc_lower) / 100.0
            scores.append(score)
        
        return scores
    
    def retrieve_code(self, query: str, k: int = 8) -> List[RetrievalResult]:
        """Retrieve code chunks."""
        if not self.code_collection:
            return []
        
        query_embedding = self.embed_provider.embed([query])[0]
        
        results = self.code_collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0
            dense_score = 1.0 - distance
            
            doc = self.code_collection.get(ids=[doc_id])
            if doc["documents"]:
                content = doc["documents"][0]
            else:
                continue
            
            bm25_scores = self._simple_bm25(query, [content])
            bm25_score = bm25_scores[0] if bm25_scores else 0.0
            
            hybrid_score = (self.dense_weight * dense_score + 
                          self.bm25_weight * bm25_score)
            
            chunks.append(RetrievalResult(
                content=content,
                metadata=metadata,
                score=hybrid_score,
                source_type="code"
            ))
        
        if self.cross_encoder and len(chunks) > k:
            query_doc_pairs = [(query, chunk.content) for chunk in chunks]
            rerank_scores = self.cross_encoder.predict(query_doc_pairs)
            
            for i, chunk in enumerate(chunks):
                chunk.score = 0.7 * chunk.score + 0.3 * rerank_scores[i]
            
            chunks.sort(key=lambda x: x.score, reverse=True)
        
        return chunks[:k]
    
    def retrieve_docs(self, query: str, k: int = 8) -> List[RetrievalResult]:
        """Retrieve document chunks."""
        if not self.docs_collection:
            return []
        
        query_embedding = self.embed_provider.embed([query])[0]
        
        results = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0
            dense_score = 1.0 - distance
            
            doc = self.docs_collection.get(ids=[doc_id])
            if doc["documents"]:
                content = doc["documents"][0]
            else:
                continue
            
            bm25_scores = self._simple_bm25(query, [content])
            bm25_score = bm25_scores[0] if bm25_scores else 0.0
            
            hybrid_score = (self.dense_weight * dense_score + 
                          self.bm25_weight * bm25_score)
            
            chunks.append(RetrievalResult(
                content=content,
                metadata=metadata,
                score=hybrid_score,
                source_type="docs"
            ))
        
        if self.cross_encoder and len(chunks) > k:
            query_doc_pairs = [(query, chunk.content) for chunk in chunks]
            rerank_scores = self.cross_encoder.predict(query_doc_pairs)
            
            for i, chunk in enumerate(chunks):
                chunk.score = 0.7 * chunk.score + 0.3 * rerank_scores[i]
            
            chunks.sort(key=lambda x: x.score, reverse=True)
        
        return chunks[:k]
    
    def retrieve_hybrid(self, query: str, k_code: int = 6, k_docs: int = 6) -> List[RetrievalResult]:
        """Retrieve from both code and docs."""
        code_results = self.retrieve_code(query, k_code)
        docs_results = self.retrieve_docs(query, k_docs)
        
        all_results = code_results + docs_results
        
        seen = set()
        unique_results = []
        for result in all_results:
            content_hash = hash(result.content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results


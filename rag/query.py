#!/usr/bin/env python3
"""Query the RAG index."""
import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from retriever import HybridRetriever
from llm import get_provider
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def format_attributions(results: List[Any], source_type: str) -> str:
    """Format attribution citations."""
    citations = []
    for i, result in enumerate(results, 1):
        meta = result.metadata
        if source_type == "code":
            citations.append(
                f"[{i}] {meta.get('path', 'unknown')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
            )
        else:
            citations.append(
                f"[{i}] {meta.get('pdf_path', 'unknown')} (pages {meta.get('page_start', '?')}-{meta.get('page_end', '?')})"
            )
    return "\n".join(citations)


def ask_code(query: str, k: int, config: Optional[Dict[str, Any]] = None) -> str:
    """Query code index."""
    if config is None:
        config = load_config()
    index_dir = config.get("index_dir", ".rag_index")
    retriever = HybridRetriever(index_dir, config)
    
    results = retriever.retrieve_code(query, k)
    
    if not results:
        return "No code chunks found. Please ensure the index has been built."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        meta = result.metadata
        context_parts.append(
            f"--- Code chunk {i} ({meta.get('path', 'unknown')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}) ---\n"
            f"{result.content}\n"
        )
    
    context = "\n".join(context_parts)
    
    provider = get_provider(
        provider=config.get("provider", "auto"),
        model=config.get("model")
    )
    
    system_prompt = """You are a code analysis assistant. Answer questions about code based on the provided context.
Always cite your sources using the format [N] where N is the chunk number.
Be precise and cite specific file paths and line numbers."""
    
    messages = [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based on the context above. Always include a 'Sources' section at the end listing all cited chunks with their file paths and line ranges."
        }
    ]
    
    answer = provider.chat(messages, system_prompt=system_prompt)
    sources = format_attributions(results, "code")
    return f"{answer}\n\n---\n\nSources:\n{sources}"


def ask_docs(query: str, k: int, config: Optional[Dict[str, Any]] = None) -> str:
    """Query docs index."""
    if config is None:
        config = load_config()
    index_dir = config.get("index_dir", ".rag_index")
    retriever = HybridRetriever(index_dir, config)
    
    results = retriever.retrieve_docs(query, k)
    
    if not results:
        return "No document chunks found. Please ensure the index has been built."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        meta = result.metadata
        context_parts.append(
            f"--- Document chunk {i} ({meta.get('pdf_path', 'unknown')}, pages {meta.get('page_start', '?')}-{meta.get('page_end', '?')}) ---\n"
            f"{result.content}\n"
        )
    
    context = "\n".join(context_parts)
    
    provider = get_provider(
        provider=config.get("provider", "auto"),
        model=config.get("model")
    )
    
    system_prompt = """You are a document analysis assistant. Answer questions about documents based on the provided context.
Always cite your sources using the format [N] where N is the chunk number.
Be precise and cite specific PDF paths and page numbers."""
    
    messages = [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based on the context above. Always include a 'Sources' section at the end listing all cited chunks with their PDF paths and page ranges."
        }
    ]
    
    answer = provider.chat(messages, system_prompt=system_prompt)
    
    sources = format_attributions(results, "docs")
    return f"{answer}\n\n---\n\nSources:\n{sources}"


def ask_combined(query: str, k_code: int, k_docs: int, config: Optional[Dict[str, Any]] = None) -> str:
    """Query both code and docs."""
    if config is None:
        config = load_config()
    index_dir = config.get("index_dir", ".rag_index")
    retriever = HybridRetriever(index_dir, config)
    
    results = retriever.retrieve_hybrid(query, k_code, k_docs)
    
    if not results:
        return "No chunks found. Please ensure the index has been built."
    
    context_parts = []
    code_citations = []
    docs_citations = []
    
    for i, result in enumerate(results, 1):
        meta = result.metadata
        if result.source_type == "code":
            context_parts.append(
                f"--- Code chunk {i} ({meta.get('path', 'unknown')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}) ---\n"
                f"{result.content}\n"
            )
            code_citations.append(
                f"[{i}] {meta.get('path', 'unknown')}:{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
            )
        else:
            context_parts.append(
                f"--- Document chunk {i} ({meta.get('pdf_path', 'unknown')}, pages {meta.get('page_start', '?')}-{meta.get('page_end', '?')}) ---\n"
                f"{result.content}\n"
            )
            docs_citations.append(
                f"[{i}] {meta.get('pdf_path', 'unknown')} (pages {meta.get('page_start', '?')}-{meta.get('page_end', '?')})"
            )
    
    context = "\n".join(context_parts)
    
    provider = get_provider(
        provider=config.get("provider", "auto"),
        model=config.get("model")
    )
    
    system_prompt = """You are a code and document analysis assistant. Answer questions based on the provided context from both code and documents.
Always cite your sources using the format [N] where N is the chunk number.
Be precise and cite specific file paths, line numbers, PDF paths, and page numbers."""
    
    messages = [
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based on the context above. Always include a 'Sources' section at the end listing all cited chunks with their file paths/line ranges or PDF paths/page ranges."
        }
    ]
    
    answer = provider.chat(messages, system_prompt=system_prompt)
    
    sources_parts = []
    if code_citations:
        sources_parts.append("Code Sources:\n" + "\n".join(code_citations))
    if docs_citations:
        sources_parts.append("Document Sources:\n" + "\n".join(docs_citations))
    
    return f"{answer}\n\n---\n\nSources:\n" + "\n\n".join(sources_parts)


def main():
    parser = argparse.ArgumentParser(description="Query RAG index")
    parser.add_argument("--type", choices=["code", "docs", "combined"], required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--k-code", type=int, default=6)
    parser.add_argument("--k-docs", type=int, default=6)
    parser.add_argument("--config", default="config.yaml")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    try:
        if args.type == "code":
            result = ask_code(args.query, args.k, config)
        elif args.type == "docs":
            result = ask_docs(args.query, args.k, config)
        else:
            result = ask_combined(args.query, args.k_code, args.k_docs, config)
        
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


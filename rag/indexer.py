#!/usr/bin/env python3
"""Index code and documents for RAG."""
import os
import sys
import argparse
import yaml
import glob
import fnmatch
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Error: chromadb not installed")
    sys.exit(1)

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("Warning: pypdf not available, PDF indexing will fail")

from chunking import CodeChunker, DocChunker, CodeChunk, DocChunk
from symbols import SymbolTable, Symbol
from callgraph import CallGraph
from llm import get_embed_provider


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def should_ignore(path: str, ignore_globs: List[str]) -> bool:
    """Check if path should be ignored."""
    for pattern in ignore_globs:
        if fnmatch.fnmatch(path, pattern) or pattern in path:
            return True
    return False


def index_code(
    repo_path: str,
    index_dir: str,
    config: Dict[str, Any],
    reset: bool = False
):
    """Index code repository."""
    print(f"Indexing code from {repo_path}...")
    
    if not os.path.exists(repo_path):
        print(f"Error: Repository path {repo_path} does not exist")
        return
    
    chunker = CodeChunker(
        max_tokens=config.get("max_chunk_tokens", 1000),
        overlap_tokens=config.get("overlap_tokens", 200)
    )
    
    symbol_table = SymbolTable(index_dir)
    if reset:
        symbol_table.clear()
    
    embed_provider = get_embed_provider(
        provider=config.get("provider", "auto"),
        embed_model=config.get("embed_model")
    )
    
    chroma_client = chromadb.PersistentClient(
        path=os.path.join(index_dir, "chroma"),
        settings=Settings(anonymized_telemetry=False)
    )
    
    if reset:
        try:
            chroma_client.delete_collection("code_index")
        except:
            pass
    
    try:
        code_collection = chroma_client.get_collection("code_index")
    except:
        code_collection = chroma_client.create_collection(
            name="code_index",
            metadata={"description": "Code chunks"}
        )
    
    code_globs = config.get("code_globs", ["**/*.{ts,tsx,js,jsx,py,go,java,rb,rs,cpp,c,cs}"])
    ignore_globs = config.get("ignore_globs", [])
    
    all_files = []
    for pattern in code_globs:
        if "{" in pattern and "}" in pattern:
            match = re.match(r"(.+)\{([^}]+)\}(.*)", pattern)
            if match:
                base, extensions, suffix = match.groups()
                for ext in extensions.split(","):
                    expanded_pattern = base + ext.strip() + suffix
                    full_pattern = os.path.join(repo_path, expanded_pattern)
                    matches = glob.glob(full_pattern, recursive=True)
                    all_files.extend(matches)
            else:
                full_pattern = os.path.join(repo_path, pattern)
                matches = glob.glob(full_pattern, recursive=True)
                all_files.extend(matches)
        else:
            full_pattern = os.path.join(repo_path, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            all_files.extend(matches)
    
    unique_files = list(set(all_files))
    files_to_index = [
        f for f in unique_files
        if os.path.isfile(f) and not should_ignore(f, ignore_globs)
    ]
    
    print(f"Found {len(files_to_index)} files to index")
    
    all_chunks = []
    file_contents = {}
    total_chunks = 0
    
    for i, file_path in enumerate(files_to_index):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files_to_index)} files...")
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_contents[file_path] = content
            chunks = chunker.chunk_file(file_path, content)
            
            for chunk in chunks:
                for symbol_name in chunk.symbols:
                    symbol = Symbol(
                        name=symbol_name,
                        file_path=chunk.path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        symbol_type="function",  # Simplified
                        language=chunk.language
                    )
                    symbol_table.add_symbol(symbol)
            
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
        
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
    
    print(f"Generated {total_chunks} chunks")
    print("Computing embeddings...")
    
    batch_size = 100
    chunk_texts = [chunk.content for chunk in all_chunks]
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_chunks = all_chunks[i:i + batch_size]
        
        try:
            embeddings = embed_provider.embed(batch)
            ids = [f"{chunk.path}:{chunk.start_line}:{chunk.end_line}:{j}" for j, chunk in enumerate(batch_chunks)]
            metadatas = [
                {
                    "path": chunk.path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "symbols": ",".join(chunk.symbols)
                }
                for chunk in batch_chunks
            ]
            
            code_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=batch,
                metadatas=metadatas
            )
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Indexed {i + batch_size}/{len(chunk_texts)} chunks...")
        
        except Exception as e:
            print(f"Warning: Failed to embed batch {i}: {e}")
    
    symbol_table.save()
    
    print("Building call graph...")
    call_graph = CallGraph(index_dir, symbol_table)
    if reset:
        call_graph.edges = []
    call_graph.build_from_files(file_contents)
    call_graph.save()
    
    print(f"✓ Indexed {total_chunks} code chunks from {len(files_to_index)} files")


def index_docs(
    docs_path: str,
    index_dir: str,
    config: Dict[str, Any],
    reset: bool = False
):
    """Index PDF documents."""
    print(f"Indexing documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        print(f"Error: Documents path {docs_path} does not exist")
        return
    
    if not PYPDF_AVAILABLE:
        print("Error: pypdf not installed, cannot index PDFs")
        return
    
    chunker = DocChunker(
        max_tokens=config.get("max_chunk_tokens", 1000),
        overlap_tokens=config.get("overlap_tokens", 200)
    )
    
    embed_provider = get_embed_provider(
        provider=config.get("provider", "auto"),
        embed_model=config.get("embed_model")
    )
    
    chroma_client = chromadb.PersistentClient(
        path=os.path.join(index_dir, "chroma"),
        settings=Settings(anonymized_telemetry=False)
    )
    
    if reset:
        try:
            chroma_client.delete_collection("docs_index")
        except:
            pass
    
    try:
        docs_collection = chroma_client.get_collection("docs_index")
    except:
        docs_collection = chroma_client.create_collection(
            name="docs_index",
            metadata={"description": "Document chunks"}
        )
    
    pdf_files = glob.glob(os.path.join(docs_path, "**/*.pdf"), recursive=True)
    print(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    
    for i, pdf_path in enumerate(pdf_files):
        print(f"Processing {i + 1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        
        try:
            reader = PdfReader(pdf_path)
            pages = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages.append((page_num, text))
            
            if not pages:
                print(f"Warning: No text extracted from {pdf_path}")
                continue
            
            chunks = chunker.chunk_pdf(pdf_path, pages)
            all_chunks.extend(chunks)
        
        except Exception as e:
            print(f"Warning: Failed to process {pdf_path}: {e}")
    
    print(f"Generated {len(all_chunks)} document chunks")
    print("Computing embeddings...")
    
    batch_size = 100
    chunk_texts = [chunk.content for chunk in all_chunks]
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_chunks = all_chunks[i:i + batch_size]
        
        try:
            embeddings = embed_provider.embed(batch)
            
            ids = [
                f"{chunk.pdf_path}:{chunk.page_start}:{chunk.page_end}:{i}"
                for i, chunk in enumerate(batch_chunks)
            ]
            metadatas = [
                {
                    "pdf_path": chunk.pdf_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end
                }
                for chunk in batch_chunks
            ]
            
            docs_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=batch,
                metadatas=metadatas
            )
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Indexed {i + batch_size}/{len(chunk_texts)} chunks...")
        
        except Exception as e:
            print(f"Warning: Failed to embed batch {i}: {e}")
    
    print(f"✓ Indexed {len(all_chunks)} document chunks from {len(pdf_files)} PDFs")


def main():
    parser = argparse.ArgumentParser(description="Index code and documents")
    parser.add_argument("--repo", help="Repository path", default="../repo")
    parser.add_argument("--docs", help="Documents path", default="../docs")
    parser.add_argument("--config", help="Config file", default="config.yaml")
    parser.add_argument("--reset", action="store_true", help="Reset index")
    parser.add_argument("--only", choices=["code", "docs"], help="Index only code or docs")
    
    args = parser.parse_args()
    
    config_paths = [
        args.config,
        os.path.join(os.path.dirname(__file__), args.config),
        "config.yaml",
        os.path.join(os.path.dirname(__file__), "config.yaml"),
    ]
    config = {}
    for cp in config_paths:
        if os.path.exists(cp):
            config = load_config(cp)
            break
    if not config:
        config = load_config(args.config)
    index_dir = config.get("index_dir", ".rag_index")
    os.makedirs(index_dir, exist_ok=True)
    
    if args.only != "docs":
        index_code(args.repo, index_dir, config, reset=args.reset)
    
    if args.only != "code":
        index_docs(args.docs, index_dir, config, reset=args.reset)
    
    print("✓ Indexing complete!")


if __name__ == "__main__":
    main()


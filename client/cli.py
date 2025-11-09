#!/usr/bin/env python3
"""CLI client for code analyser."""
import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))


async def call_tool(session: Any, tool_name: str, arguments: Dict[str, Any]) -> str:
    from rag.query import ask_code, ask_docs, ask_combined
    from rag.symbols import SymbolTable
    from rag.callgraph import CallGraph
    from rag.indexer import load_config
    
    config = load_config()
    
    if tool_name == "ask_code":
        return ask_code(arguments.get("query"), arguments.get("k", 8), config)
    elif tool_name == "ask_docs":
        return ask_docs(arguments.get("query"), arguments.get("k", 8), config)
    elif tool_name == "ask_combined":
        return ask_combined(
            arguments.get("query"),
            arguments.get("k_code", 6),
            arguments.get("k_docs", 6),
            config
        )
    elif tool_name == "search_symbols":
        index_dir = config.get("index_dir", ".rag_index")
        symbol_table = SymbolTable(index_dir)
        results = symbol_table.search(arguments.get("q"), limit=20)
        if results:
            formatted = "\n".join(
                f"- {sym.name} ({sym.symbol_type}) at {sym.file_path}:{sym.start_line}"
                for sym, score in results[:20]
            )
            return f'Found {len(results)} symbol(s):\n\n{formatted}'
        return f'No symbols found matching "{arguments.get("q")}"'
    elif tool_name == "trace_calls":
        index_dir = config.get("index_dir", ".rag_index")
        symbol_table = SymbolTable(index_dir)
        call_graph = CallGraph(index_dir, symbol_table)
        traces = call_graph.trace_calls(arguments.get("symbol"))
        result = f'Call graph for "{arguments.get("symbol")}":\n\n'
        if traces["inbound"]:
            result += f'Inbound calls ({len(traces["inbound"])}):\n'
            for edge in traces["inbound"][:10]:
                result += f'  - {edge.caller} → {edge.callee} ({edge.caller_file}:{edge.caller_line})\n'
            result += "\n"
        if traces["outbound"]:
            result += f'Outbound calls ({len(traces["outbound"])}):\n'
            for edge in traces["outbound"][:10]:
                result += f'  - {edge.caller} → {edge.callee} ({edge.callee_file}:{edge.callee_line})\n'
        return result
    else:
        return f"Unknown tool: {tool_name}"


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Code Analyser CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    parser_code = subparsers.add_parser("ask:code", help="Ask questions about code")
    parser_code.add_argument("query", nargs="+", help="Question to ask")
    parser_code.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve")
    
    parser_docs = subparsers.add_parser("ask:docs", help="Ask questions about documents")
    parser_docs.add_argument("query", nargs="+", help="Question to ask")
    parser_docs.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve")
    
    parser_both = subparsers.add_parser("ask:both", help="Ask questions using both code and docs")
    parser_both.add_argument("query", nargs="+", help="Question to ask")
    parser_both.add_argument("--k-code", type=int, default=6, help="Number of code chunks")
    parser_both.add_argument("--k-docs", type=int, default=6, help="Number of doc chunks")
    
    parser_symbols = subparsers.add_parser("symbols", help="Search for symbols")
    parser_symbols.add_argument("name", nargs="+", help="Symbol name to search")
    
    parser_trace = subparsers.add_parser("trace", help="Trace call graph")
    parser_trace.add_argument("symbol", nargs="+", help="Symbol name to trace")
    
    parser_reindex = subparsers.add_parser("reindex", help="Rebuild index")
    parser_reindex.add_argument("--only", choices=["code", "docs"], help="Index only code or docs")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "ask:code":
        query = " ".join(args.query)
        result = await call_tool(None, "ask_code", {"query": query, "k": args.k})
        print("\n" + result + "\n")
    
    elif args.command == "ask:docs":
        query = " ".join(args.query)
        result = await call_tool(None, "ask_docs", {"query": query, "k": args.k})
        print("\n" + result + "\n")
    
    elif args.command == "ask:both":
        query = " ".join(args.query)
        result = await call_tool(None, "ask_combined", {
            "query": query,
            "k_code": args.k_code,
            "k_docs": args.k_docs
        })
        print("\n" + result + "\n")
    
    elif args.command == "symbols":
        name = " ".join(args.name)
        result = await call_tool(None, "search_symbols", {"q": name})
        print("\n" + result + "\n")
    
    elif args.command == "trace":
        symbol = " ".join(args.symbol)
        result = await call_tool(None, "trace_calls", {"symbol": symbol})
        print("\n" + result + "\n")
    
    elif args.command == "reindex":
        from rag.indexer import index_code, index_docs, load_config
        config = load_config()
        target = args.only or "all"
        repo_path = config.get("repo_path", "../repo")
        docs_path = config.get("docs_path", "../docs")
        index_dir = config.get("index_dir", ".rag_index")
        
        if target in ["code", "all"]:
            print("Indexing code...")
            index_code(repo_path, index_dir, config, reset=False)
        if target in ["docs", "all"]:
            print("Indexing documents...")
            index_docs(docs_path, index_dir, config, reset=False)
        print("✓ Indexing complete!")


if __name__ == "__main__":
    asyncio.run(main())


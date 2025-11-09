#!/usr/bin/env python3
"""MCP server for code and document analysis."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.query import ask_code, ask_docs, ask_combined
from rag.symbols import SymbolTable
from rag.callgraph import CallGraph
from rag.indexer import index_code, index_docs, load_config


class CodeAnalyserServer:
    """MCP server for code analysis."""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        possible_paths = [
            Path(__file__).parent.parent / "rag" / "config.yaml",
            Path.cwd() / "rag" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        
        for config_path in possible_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"Warning: Could not load {config_path}: {e}", file=sys.stderr)
        
        return {}
    
    def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> str:
        """Handle tool calls."""
        try:
            if name == "ask_code":
                query = arguments.get("query")
                k = arguments.get("k", self.config.get("k_code", 8))
                return ask_code(query, k, self.config)
            
            elif name == "ask_docs":
                query = arguments.get("query")
                k = arguments.get("k", self.config.get("k_docs", 8))
                return ask_docs(query, k, self.config)
            
            elif name == "ask_combined":
                query = arguments.get("query")
                k_code = arguments.get("k_code", 6)
                k_docs = arguments.get("k_docs", 6)
                return ask_combined(query, k_code, k_docs, self.config)
            
            elif name == "search_symbols":
                q = arguments.get("q")
                index_dir = self.config.get("index_dir", ".rag_index")
                symbol_table = SymbolTable(index_dir)
                results = symbol_table.search(q, limit=20)
                
                if not results:
                    return f'No symbols found matching "{q}"'
                
                formatted = "\n".join(
                    f"- {sym.name} ({sym.symbol_type}) at {sym.file_path}:{sym.start_line}"
                    for sym, score in results[:20]
                )
                return f'Found {len(results)} symbol(s) matching "{q}":\n\n{formatted}'
            
            elif name == "trace_calls":
                symbol = arguments.get("symbol")
                index_dir = self.config.get("index_dir", ".rag_index")
                symbol_table = SymbolTable(index_dir)
                call_graph = CallGraph(index_dir, symbol_table)
                traces = call_graph.trace_calls(symbol)
                
                result = f'Call graph for "{symbol}":\n\n'
                
                if traces["inbound"]:
                    result += f'Inbound calls ({len(traces["inbound"])}):\n'
                    for edge in traces["inbound"][:10]:
                        result += f'  - {edge.caller} → {edge.callee} ({edge.caller_file}:{edge.caller_line})\n'
                    result += "\n"
                else:
                    result += "No inbound calls found.\n\n"
                
                if traces["outbound"]:
                    result += f'Outbound calls ({len(traces["outbound"])}):\n'
                    for edge in traces["outbound"][:10]:
                        result += f'  - {edge.caller} → {edge.callee} ({edge.callee_file}:{edge.callee_line})\n'
                else:
                    result += "No outbound calls found.\n"
                
                return result
            
            elif name == "update_index":
                target = arguments.get("target", "all")
                repo_path = self.config.get("repo_path", "../repo")
                docs_path = self.config.get("docs_path", "../docs")
                index_dir = self.config.get("index_dir", ".rag_index")
                
                output = []
                if target in ["code", "all"]:
                    output.append("Indexing code...\n")
                    index_code(repo_path, index_dir, self.config, reset=False)
                    output.append("✓ Code indexing complete\n")
                
                if target in ["docs", "all"]:
                    output.append("Indexing documents...\n")
                    index_docs(docs_path, index_dir, self.config, reset=False)
                    output.append("✓ Document indexing complete\n")
                
                return "".join(output)
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def run(self):
        """Run the MCP server using JSON-RPC over stdio."""
        await self._run_fallback()
    
    async def _run_fallback(self):
        """Fallback implementation using basic JSON-RPC."""
        import sys
        
        async def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
            """Handle JSON-RPC request."""
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            try:
                if method == "tools/list":
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {"name": "ask_code", "description": "Ask questions about code"},
                                {"name": "ask_docs", "description": "Ask questions about documents"},
                                {"name": "ask_combined", "description": "Ask questions using both"},
                                {"name": "search_symbols", "description": "Search for symbols"},
                                {"name": "trace_calls", "description": "Trace call graph"},
                                {"name": "update_index", "description": "Update index"},
                            ]
                        }
                    }
                
                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    
                    result = self._handle_tool_call(tool_name, arguments)
                    
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{"type": "text", "text": result}]
                        }
                    }
                
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"}
                    }
            
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(e)}
                }
        
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = await handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if 'request' in locals() else None,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                }
                print(json.dumps(error_response), flush=True)


async def main():
    """Main entry point."""
    server = CodeAnalyserServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())


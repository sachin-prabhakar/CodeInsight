"""Call graph construction and traversal."""
import os
import json
import re
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
try:
    from symbols import SymbolTable, Symbol
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from symbols import SymbolTable, Symbol

@dataclass
class CallEdge:
    """Represents a call relationship."""
    caller: str  # Symbol name
    callee: str  # Symbol name
    caller_file: str
    caller_line: int
    callee_file: str
    callee_line: int


class CallGraph:
    """Manages call graph for code analysis."""
    
    def __init__(self, index_dir: str, symbol_table: SymbolTable):
        self.index_dir = index_dir
        self.symbol_table = symbol_table
        self.graph_file = os.path.join(index_dir, "callgraph.json")
        self.edges: List[CallEdge] = []
        self._load()
    
    def _load(self):
        """Load call graph from disk."""
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, "r") as f:
                    data = json.load(f)
                    self.edges = [
                        CallEdge(**edge) for edge in data.get("edges", [])
                    ]
            except Exception as e:
                print(f"Warning: Failed to load call graph: {e}")
                self.edges = []
    
    def save(self):
        """Save call graph to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        data = {
            "edges": [
                {
                    "caller": e.caller,
                    "callee": e.callee,
                    "caller_file": e.caller_file,
                    "caller_line": e.caller_line,
                    "callee_file": e.callee_file,
                    "callee_line": e.callee_line
                }
                for e in self.edges
            ]
        }
        
        with open(self.graph_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_edge(self, edge: CallEdge):
        """Add a call edge."""
        self.edges.append(edge)
    
    def trace_calls(self, symbol_name: str) -> Dict[str, List[CallEdge]]:
        """Trace inbound and outbound calls for a symbol."""
        inbound = []
        outbound = []
        
        for edge in self.edges:
            if edge.callee == symbol_name:
                inbound.append(edge)
            if edge.caller == symbol_name:
                outbound.append(edge)
        
        return {
            "inbound": inbound,
            "outbound": outbound
        }
    
    def build_from_files(self, file_contents: Dict[str, str]):
        """Build call graph by analyzing file contents (best-effort)."""
        self.edges = []
        
        all_symbols: Dict[str, Symbol] = {}
        for name, syms in self.symbol_table.symbols.items():
            for sym in syms:
                key = f"{sym.file_path}:{sym.start_line}"
                all_symbols[key] = sym
        
        for file_path, content in file_contents.items():
            lines = content.split("\n")
            
            file_symbols = [
                sym for sym in all_symbols.values()
                if sym.file_path == file_path
            ]
            
            for i, line in enumerate(lines, 1):
                matches = re.finditer(r'(\w+)\s*\(', line)
                for match in matches:
                    callee_name = match.group(1)
                    
                    for callee_sym in self.symbol_table.get_symbol(callee_name):
                        for caller_sym in file_symbols:
                            if caller_sym.start_line <= i <= caller_sym.end_line:
                                edge = CallEdge(
                                    caller=caller_sym.name,
                                    callee=callee_sym.name,
                                    caller_file=caller_sym.file_path,
                                    caller_line=i,
                                    callee_file=callee_sym.file_path,
                                    callee_line=callee_sym.start_line
                                )
                                self.add_edge(edge)
                                break


"""Symbol table and search utilities."""
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from rapidfuzz import fuzz, process

@dataclass
class Symbol:
    """Represents a code symbol."""
    name: str
    file_path: str
    start_line: int
    end_line: int
    symbol_type: str  # function, class, method, etc.
    language: str


class SymbolTable:
    """Manages symbol table for code search."""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.symbols_file = os.path.join(index_dir, "symbols.json")
        self.symbols: Dict[str, List[Symbol]] = {}  # name -> [Symbol]
        self._load()
    
    def _load(self):
        """Load symbol table from disk."""
        if os.path.exists(self.symbols_file):
            try:
                with open(self.symbols_file, "r") as f:
                    data = json.load(f)
                    self.symbols = {}
                    for name, syms in data.items():
                        self.symbols[name] = [
                            Symbol(**sym) for sym in syms
                        ]
            except Exception as e:
                print(f"Warning: Failed to load symbol table: {e}")
                self.symbols = {}
    
    def save(self):
        """Save symbol table to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        data = {}
        for name, syms in self.symbols.items():
            data[name] = [
                {
                    "name": s.name,
                    "file_path": s.file_path,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "symbol_type": s.symbol_type,
                    "language": s.language
                }
                for s in syms
            ]
        
        with open(self.symbols_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_symbol(self, symbol: Symbol):
        """Add a symbol to the table."""
        if symbol.name not in self.symbols:
            self.symbols[symbol.name] = []
        self.symbols[symbol.name].append(symbol)
    
    def search(self, query: str, limit: int = 20) -> List[Tuple[Symbol, float]]:
        """Search for symbols using fuzzy matching."""
        if not self.symbols:
            return []
        
        all_names = list(self.symbols.keys())
        
        matches = process.extract(
            query,
            all_names,
            scorer=fuzz.partial_ratio,
            limit=limit
        )
        
        results = []
        for name, score, _ in matches:
            for symbol in self.symbols[name]:
                results.append((symbol, score / 100.0))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_symbol(self, name: str) -> List[Symbol]:
        """Get all symbols with exact name match."""
        return self.symbols.get(name, [])
    
    def clear(self):
        """Clear all symbols."""
        self.symbols = {}


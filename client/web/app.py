#!/usr/bin/env python3
"""Web server for code analyser."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rag.query import ask_code, ask_docs, ask_combined
from rag.symbols import SymbolTable
from rag.callgraph import CallGraph
from rag.indexer import load_config

app = FastAPI(title="Code Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ui_path = Path(__file__).parent / "ui"
if ui_path.exists():
    app.mount("/static", StaticFiles(directory=str(ui_path)), name="static")


class QueryRequest(BaseModel):
    query: str
    type: str
    sessionId: Optional[str] = None


class SymbolRequest(BaseModel):
    q: str
    sessionId: Optional[str] = None


class TraceRequest(BaseModel):
    symbol: str
    sessionId: Optional[str] = None


config_paths = [
    project_root / "rag" / "config.yaml",
    project_root / "config.yaml",
]
config = {}
for cp in config_paths:
    if cp.exists():
        config = load_config(str(cp))
        break
if not config:
    config = load_config()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    index_file = ui_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Code Analyser API</h1><p>Web UI not found. Use /api endpoints.</p>")


@app.post("/api/query")
async def query(request: QueryRequest):
    """Handle query requests."""
    try:
        if request.type == "code":
            result = ask_code(request.query, 8, config)
        elif request.type == "docs":
            result = ask_docs(request.query, 8, config)
        elif request.type == "both":
            result = ask_combined(request.query, 6, 6, config)
        else:
            raise HTTPException(status_code=400, detail="Invalid type")
        
        return {"answer": result, "type": request.type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/symbols")
async def symbols(request: SymbolRequest):
    """Handle symbol search requests."""
    try:
        index_dir = config.get("index_dir", ".rag_index")
        symbol_table = SymbolTable(index_dir)
        results = symbol_table.search(request.q, limit=20)
        
        if not results:
            return {"result": f'No symbols found matching "{request.q}"'}
        
        formatted = "\n".join(
            f"- {sym.name} ({sym.symbol_type}) at {sym.file_path}:{sym.start_line}"
            for sym, score in results[:20]
        )
        return {"result": f'Found {len(results)} symbol(s):\n\n{formatted}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trace")
async def trace(request: TraceRequest):
    """Handle call trace requests."""
    try:
        index_dir = config.get("index_dir", ".rag_index")
        symbol_table = SymbolTable(index_dir)
        call_graph = CallGraph(index_dir, symbol_table)
        traces = call_graph.trace_calls(request.symbol)
        
        result = f'Call graph for "{request.symbol}":\n\n'
        if traces["inbound"]:
            result += f'Inbound calls ({len(traces["inbound"])}):\n'
            for edge in traces["inbound"][:10]:
                result += f'  - {edge.caller} → {edge.callee} ({edge.caller_file}:{edge.caller_line})\n'
            result += "\n"
        if traces["outbound"]:
            result += f'Outbound calls ({len(traces["outbound"])}):\n'
            for edge in traces["outbound"][:10]:
                result += f'  - {edge.caller} → {edge.callee} ({edge.callee_file}:{edge.callee_line})\n'
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    uvicorn.run(app, host="0.0.0.0", port=port)


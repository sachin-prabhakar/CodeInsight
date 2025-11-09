.PHONY: setup index index-code index-docs mcp cli web demo clean help


all: setup

setup:
	@echo "Setting up Python environment..."
	python3 -m venv venv || python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r rag/requirements.txt
	@echo "✓ Setup complete!"

index:
	@echo "Indexing code and documents..."
	. venv/bin/activate && python rag/indexer.py --repo repo --docs docs
	@echo "✓ Indexing complete"

index-code:
	@echo "Indexing code..."
	. venv/bin/activate && python rag/indexer.py --repo repo --only code
	@echo "✓ Code indexing complete"

index-docs:
	@echo "Indexing documents..."
	. venv/bin/activate && python rag/indexer.py --docs docs --only docs
	@echo "✓ Document indexing complete"

mcp:
	@echo "Starting MCP server..."
	. venv/bin/activate && python mcp/server.py

cli:
	@echo "Running CLI..."
	. venv/bin/activate && python client/cli.py

web:
	@echo "Starting web UI on http://localhost:3000..."
	. venv/bin/activate && python client/web/app.py

demo:
	@echo "Running demo queries..."
	. venv/bin/activate && python client/cli.py ask:code "How does the system work?"

clean:
	rm -rf venv
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .rag_index
	@echo "✓ Clean complete"

help:
	@echo "Available targets:"
	@echo "  make setup       - Install dependencies"
	@echo "  make index       - Index code and documents"
	@echo "  make index-code  - Index code only"
	@echo "  make index-docs  - Index documents only"
	@echo "  make cli         - Run CLI"
	@echo "  make web         - Start web UI"
	@echo "  make clean       - Clean build artifacts"


# CodeInsight

AI-powered code and document analysis using local LLMs (Ollama).

**Models:** `llama3` (LLM) | `all-MiniLM-L6-v2` (embeddings)

## Quick Start

```bash
# 1. Install Ollama
brew install ollama  # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux

# 2. Start Ollama and pull model
ollama serve
ollama pull llama3

# 3. Setup
make setup
echo "OLLAMA_HOST=http://localhost:11434" > .env

# 4. Add your code/docs
mkdir -p repo docs
cp -r /path/to/code/* repo/
cp /path/to/*.pdf docs/

# 5. Index
make index

# 6. Use it
source venv/bin/activate
python client/cli.py ask:code "How does authentication work?"
```

## Usage

### CLI

```bash
source venv/bin/activate

python client/cli.py ask:code "Your question"
python client/cli.py ask:docs "Your question"
python client/cli.py ask:both "Your question"
python client/cli.py symbols "functionName"
python client/cli.py trace "functionName"
python client/cli.py reindex
```

### Web UI

```bash
source venv/bin/activate
make web
# Open http://localhost:3000
```

## config

Edit `rag/config.yaml`:

```yaml
provider: ollama
model: llama3
embed_model: local
repo_path: repo
docs_path: docs
```

# Example Queries

## Basic Queries

### Code Queries

```bash
python client/cli.py ask:code "How does the system work?"
python client/cli.py ask:code "Where is authentication handled?"
python client/cli.py ask:code "How are threads synchronized?"
```

### Document Queries

```bash
python client/cli.py ask:docs "What are the requirements?"
python client/cli.py ask:docs "What is the architecture?"
```

### Combined Queries

```bash
python client/cli.py ask:both "Does the code match the documentation?"
```

## Symbol Search

```bash
python client/cli.py symbols "functionName"
python client/cli.py symbols "ClassName"
```

## Call Graph

```bash
python client/cli.py trace "functionName"
```

## Index Management

```bash
# Rebuild index
python client/cli.py reindex

# Rebuild only code
python client/cli.py reindex --only code

# Rebuild only docs
python client/cli.py reindex --only docs
```

"""Code and document chunking utilities."""
import os
import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    from tree_sitter_python import language as python_lang
    PYTHON_LANG_AVAILABLE = True
except ImportError:
    PYTHON_LANG_AVAILABLE = False

try:
    from tree_sitter_javascript import language as js_lang
    JS_LANG_AVAILABLE = True
except ImportError:
    JS_LANG_AVAILABLE = False

try:
    from tree_sitter_typescript import language as ts_lang
    TS_LANG_AVAILABLE = True
except ImportError:
    TS_LANG_AVAILABLE = False

try:
    from tree_sitter_go import language as go_lang
    GO_LANG_AVAILABLE = True
except ImportError:
    GO_LANG_AVAILABLE = False

try:
    from tree_sitter_java import language as java_lang
    JAVA_LANG_AVAILABLE = True
except ImportError:
    JAVA_LANG_AVAILABLE = False

try:
    from tree_sitter_ruby import language as ruby_lang
    RUBY_LANG_AVAILABLE = True
except ImportError:
    RUBY_LANG_AVAILABLE = False

try:
    from tree_sitter_rust import language as rust_lang
    RUST_LANG_AVAILABLE = True
except ImportError:
    RUST_LANG_AVAILABLE = False

try:
    from tree_sitter_cpp import language as cpp_lang
    CPP_LANG_AVAILABLE = True
except ImportError:
    CPP_LANG_AVAILABLE = False

try:
    from tree_sitter_c import language as c_lang
    C_LANG_AVAILABLE = True
except ImportError:
    C_LANG_AVAILABLE = False

try:
    from tree_sitter_c_sharp import language as csharp_lang
    CSHARP_LANG_AVAILABLE = True
except ImportError:
    CSHARP_LANG_AVAILABLE = False


@dataclass
class CodeChunk:
    """Represents a chunk of code."""
    content: str
    path: str
    start_line: int
    end_line: int
    language: str
    symbols: List[str]  # Function/class names in this chunk


@dataclass
class DocChunk:
    """Represents a chunk of document."""
    content: str
    pdf_path: str
    page_start: int
    page_end: int


class CodeChunker:
    """Chunks code files using tree-sitter when available."""
    
    def __init__(self, max_tokens: int = 1000, overlap_tokens: int = 200):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.parsers = {}
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            return
        
        lang_map = {}
        
        if PYTHON_LANG_AVAILABLE:
            lang_map["python"] = python_lang
        if JS_LANG_AVAILABLE:
            lang_map["javascript"] = js_lang
        if TS_LANG_AVAILABLE:
            lang_map["typescript"] = ts_lang
        if GO_LANG_AVAILABLE:
            lang_map["go"] = go_lang
        if JAVA_LANG_AVAILABLE:
            lang_map["java"] = java_lang
        if RUBY_LANG_AVAILABLE:
            lang_map["ruby"] = ruby_lang
        if RUST_LANG_AVAILABLE:
            lang_map["rust"] = rust_lang
        if CPP_LANG_AVAILABLE:
            lang_map["cpp"] = cpp_lang
        if C_LANG_AVAILABLE:
            lang_map["c"] = c_lang
        if CSHARP_LANG_AVAILABLE:
            lang_map["csharp"] = csharp_lang
        
        for lang_name, lang_module in lang_map.items():
            try:
                lang = Language(lang_module())
                parser = Parser()
                parser.language = lang
                self.parsers[lang_name] = parser
            except Exception as e:
                print(f"Warning: Could not initialize parser for {lang_name}, will use fallback chunking: {e}", file=sys.stderr)
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".cc": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
        }
        return ext_map.get(ext)
    
    def _extract_functions_classes(self, tree, source: bytes, lang: str) -> List[Tuple[int, int, str, str]]:
        """Extract function and class definitions from tree-sitter AST."""
        results = []
        
        def traverse(node, depth=0):
            node_type = node.type
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            name = None
            for child in node.children:
                if child.type in ["identifier", "property_identifier", "type_identifier"]:
                    name = source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                    break
            
            if lang == "python":
                if node_type in ["function_definition", "class_definition"]:
                    if name:
                        results.append((start_line, end_line, node_type, name))
            elif lang in ["javascript", "typescript"]:
                if node_type in ["function_declaration", "method_definition", "class_declaration"]:
                    if name:
                        results.append((start_line, end_line, node_type, name))
            elif lang == "go":
                if node_type in ["function_declaration", "method_declaration", "type_declaration"]:
                    if name:
                        results.append((start_line, end_line, node_type, name))
            elif lang == "java":
                if node_type in ["method_declaration", "class_declaration", "interface_declaration"]:
                    if name:
                        results.append((start_line, end_line, node_type, name))
            elif lang == "rust":
                if node_type in ["function_item", "impl_item", "struct_item", "trait_item"]:
                    if name:
                        results.append((start_line, end_line, node_type, name))
            elif lang == "ruby":
                if node_type in ["method", "class", "module"]:
                    if name:
                        results.append((start_line, end_line, node_type, name))
            elif lang in ["cpp", "c", "csharp"]:
                if node_type in ["function_definition", "class_specifier", "method_definition", "function_declarator"]:
                    if not name:
                        for child in node.children:
                            if child.type == "function_declarator":
                                for grandchild in child.children:
                                    if grandchild.type == "identifier":
                                        name = source[grandchild.start_byte:grandchild.end_byte].decode("utf-8", errors="ignore")
                                        break
                            elif child.type == "identifier" and node_type == "function_definition":
                                name = source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                                break
                    if name:
                        results.append((start_line, end_line, node_type, name))
            
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(tree.root_node)
        return results
    
    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk a code file."""
        lang = self._detect_language(file_path)
        lines = content.split("\n")
        
        if lang and lang in self.parsers:
            try:
                parser = self.parsers[lang]
                tree = parser.parse(bytes(content, "utf-8"))
                functions_classes = self._extract_functions_classes(tree, bytes(content, "utf-8"), lang)
                
                if functions_classes:
                    chunks = []
                    functions_classes.sort(key=lambda x: x[0])
                    
                    for start_line, end_line, node_type, name in functions_classes:
                        chunk_content = "\n".join(lines[start_line - 1:end_line])
                        tokens = len(self.enc.encode(chunk_content))
                        
                        if tokens <= self.max_tokens:
                            chunks.append(CodeChunk(
                                content=chunk_content,
                                path=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                language=lang or "unknown",
                                symbols=[name]
                            ))
                        else:
                            sub_chunks = self._sliding_window_chunk(
                                chunk_content, file_path, start_line, end_line, lang
                            )
                            chunks.extend(sub_chunks)
                    
                    if lang in ["cpp", "c", "csharp"]:
                        if functions_classes and functions_classes[0][0] > 1:
                            top_level_end = functions_classes[0][0] - 1
                            if top_level_end > 0:
                                top_content = "\n".join(lines[0:top_level_end])
                                if any(keyword in top_content for keyword in ["barrier", "mutex", "semaphore", "thread", "atomic"]):
                                    top_tokens = len(self.enc.encode(top_content))
                                    if top_tokens <= self.max_tokens:
                                        chunks.insert(0, CodeChunk(
                                            content=top_content,
                                            path=file_path,
                                            start_line=1,
                                            end_line=top_level_end,
                                            language=lang,
                                            symbols=[]
                                        ))
                                    else:
                                        sub_chunks = self._sliding_window_chunk(
                                            top_content, file_path, 1, top_level_end, lang
                                        )
                                        chunks = sub_chunks + chunks
                    
                    return chunks
            except Exception as e:
                print(f"Warning: Tree-sitter parsing failed for {file_path}: {e}", file=sys.stderr)
        
        return self._sliding_window_chunk(content, file_path, 1, len(lines), lang or "unknown")
    
    def _sliding_window_chunk(
        self, content: str, path: str, start_line: int, end_line: int, lang: str
    ) -> List[CodeChunk]:
        """Fallback chunking using sliding window."""
        lines = content.split("\n")
        chunks = []
        current_chunk_lines = []
        current_start = start_line
        current_tokens = 0
        
        for i, line in enumerate(lines):
            line_tokens = len(self.enc.encode(line))
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    path=path,
                    start_line=current_start,
                    end_line=current_start + len(current_chunk_lines) - 1,
                    language=lang,
                    symbols=[]
                ))
                
                overlap_lines = []
                overlap_tokens = 0
                for j in range(len(current_chunk_lines) - 1, -1, -1):
                    line_tok = len(self.enc.encode(current_chunk_lines[j]))
                    if overlap_tokens + line_tok <= self.overlap_tokens:
                        overlap_lines.insert(0, current_chunk_lines[j])
                        overlap_tokens += line_tok
                    else:
                        break
                
                current_chunk_lines = overlap_lines
                current_start = current_start + len(current_chunk_lines) - len(overlap_lines)
                current_tokens = overlap_tokens
            
                current_chunk_lines.append(line)
            current_tokens += line_tokens
        
        if current_chunk_lines:
            chunks.append(CodeChunk(
                content="\n".join(current_chunk_lines),
                path=path,
                start_line=current_start,
                end_line=current_start + len(current_chunk_lines) - 1,
                language=lang,
                symbols=[]
            ))
        
        return chunks


class DocChunker:
    """Chunks PDF documents."""
    
    def __init__(self, max_tokens: int = 1000, overlap_tokens: int = 200):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def chunk_pdf(self, pdf_path: str, pages: List[Tuple[int, str]]) -> List[DocChunk]:
        """Chunk PDF pages with heading awareness."""
        chunks = []
        current_chunk_pages = []
        current_chunk_text = []
        current_tokens = 0
        page_start = None
        
        for page_num, page_text in pages:
            lines = page_text.split("\n")
            headings = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if (len(stripped) < 80 and 
                    (stripped.isupper() or 
                     stripped.startswith("#") or
                     (i < len(lines) - 1 and len(lines[i+1].strip()) == 0))):
                    headings.append((i, stripped))
            
            if headings:
                for heading_idx, (line_idx, heading) in enumerate(headings):
                    next_heading_idx = headings[heading_idx + 1][0] if heading_idx + 1 < len(headings) else len(lines)
                    section_lines = lines[line_idx:next_heading_idx]
                    section_text = "\n".join(section_lines)
                    section_tokens = len(self.enc.encode(section_text))
                    
                    if section_tokens <= self.max_tokens:
                        if page_start is None:
                            page_start = page_num
                        chunks.append(DocChunk(
                            content=section_text,
                            pdf_path=pdf_path,
                            page_start=page_start,
                            page_end=page_num
                        ))
                        page_start = None
                    else:
                        sub_chunks = self._sliding_window_chunk(
                            section_text, pdf_path, page_num, page_num
                        )
                        chunks.extend(sub_chunks)
            else:
                page_tokens = len(self.enc.encode(page_text))
                
                if current_tokens + page_tokens <= self.max_tokens:
                    if page_start is None:
                        page_start = page_num
                    current_chunk_text.append(page_text)
                    current_tokens += page_tokens
                else:
                    if current_chunk_text:
                        chunks.append(DocChunk(
                            content="\n\n".join(current_chunk_text),
                            pdf_path=pdf_path,
                            page_start=page_start or page_num,
                            page_end=page_num - 1
                        ))
                    
                    page_start = page_num
                    current_chunk_text = [page_text]
                    current_tokens = page_tokens
        
        if current_chunk_text:
            chunks.append(DocChunk(
                content="\n\n".join(current_chunk_text),
                pdf_path=pdf_path,
                page_start=page_start or pages[-1][0],
                page_end=pages[-1][0]
            ))
        
        return chunks
    
    def _sliding_window_chunk(self, text: str, pdf_path: str, page_start: int, page_end: int) -> List[DocChunk]:
        """Sliding window chunking for text."""
        chunks = []
        sentences = re.split(r'([.!?]\s+)', text)
        current_chunk = []
        current_tokens = 0
        current_page = page_start
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            sent_tokens = len(self.enc.encode(sentence))
            
            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                chunks.append(DocChunk(
                    content="".join(current_chunk),
                    pdf_path=pdf_path,
                    page_start=current_page,
                    page_end=current_page
                ))
                
                overlap = []
                overlap_tokens = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    tok = len(self.enc.encode(current_chunk[j]))
                    if overlap_tokens + tok <= self.overlap_tokens:
                        overlap.insert(0, current_chunk[j])
                        overlap_tokens += tok
                    else:
                        break
                
                current_chunk = overlap
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(DocChunk(
                content="".join(current_chunk),
                pdf_path=pdf_path,
                page_start=current_page,
                page_end=page_end
            ))
        
        return chunks


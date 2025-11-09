"""Provider-agnostic LLM and embedding interface."""
import os
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""
    
    def __init__(self, model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-small"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.embed_model = embed_model
    
    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate chat completion."""
        msgs = messages.copy()
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=texts
        )
        return [item.embedding for item in response.data]


class AnthropicProvider(LLMProvider):
    """Anthropic provider."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._embed_provider = None
    
    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate chat completion."""
        system_msg = system_prompt or ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "user":
                user_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                user_messages.append({"role": "assistant", "content": msg["content"]})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_msg,
            messages=user_messages
        )
        return response.content[0].text
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings - falls back to OpenAI or local."""
        if self._embed_provider is None:
            if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                self._embed_provider = OpenAIProvider()
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                self._embed_provider = LocalEmbedProvider()
            else:
                raise RuntimeError("No embedding provider available")
        return self._embed_provider.embed(texts)


class OllamaProvider(LLMProvider):
    """Ollama provider for local models."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package not installed")
        self.host = os.getenv("OLLAMA_HOST", host)
        self.model = model
        self._embed_model = None
    
    def chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Generate chat completion via Ollama."""
        try:
            ollama_messages = []
            if system_prompt:
                ollama_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    ollama_messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": ollama_messages,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            prompt = ""
            if system_prompt:
                prompt += f"System: {system_prompt}\n\n"
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            prompt += "Assistant:"
            
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()["response"]
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via Ollama."""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=120
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings


class LocalEmbedProvider:
    """Local sentence-transformers provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        import numpy as np
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        elif isinstance(embeddings, list):
            return [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
        else:
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)


def get_provider(provider: str = "auto", model: Optional[str] = None, embed_model: Optional[str] = None) -> LLMProvider:
    """Get the appropriate LLM provider based on environment and config."""
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OLLAMA_HOST"):
            provider = "ollama"
        else:
            provider = "local"
    
    if provider == "openai":
        return OpenAIProvider(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embed_model=embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        )
    elif provider == "ollama":
        return OllamaProvider(
            model=model or "llama3",
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
    else:
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        elif ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicProvider()
        else:
            raise RuntimeError(
                "No LLM provider available. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or OLLAMA_HOST environment variable."
            )


def get_embed_provider(provider: str = "auto", embed_model: Optional[str] = None):
    """Get embedding provider."""
    if embed_model == "local" or (provider == "ollama" and embed_model != "openai"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed for local embeddings")
        return LocalEmbedProvider()
    
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY") and embed_model != "local":
            return OpenAIProvider(embed_model=embed_model or "text-embedding-3-small")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            return LocalEmbedProvider()
        else:
            raise RuntimeError("No embedding provider available")
    
    if provider == "openai":
        return OpenAIProvider(embed_model=embed_model or "text-embedding-3-small")
    else:
        return LocalEmbedProvider()


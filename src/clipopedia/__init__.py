"""Clip'O'pedia — a mention-driven hybrid-RAG assistant for podcast clips.

The package is intentionally import-light: importing ``clipopedia`` pulls in no
heavy third-party libraries. Service adapters (OpenAI, Pinecone, Cohere, …) are
imported lazily, only when the "live" backend is built, so the core retrieval
logic and the offline demo run with a minimal dependency set.
"""

__version__ = "0.1.0"

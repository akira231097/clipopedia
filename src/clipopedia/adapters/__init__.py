"""Concrete implementations of the ports.

* :mod:`clipopedia.adapters.memory` — deterministic, in-process fakes (no
  network) used by the offline demo and the tests.
* The remaining modules wrap real services (OpenAI, Pinecone, Cohere, Gemini,
  SQS, X, PostgreSQL) and are imported lazily by :mod:`clipopedia.factory`.
"""

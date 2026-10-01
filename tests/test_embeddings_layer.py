"""Tests for services/embeddings: providers, vector store, chunking, RAG."""
import pytest

from services.embeddings import (
    LocalHashEmbeddingProvider,
    InMemoryVectorStore,
    semantic_chunk,
    RAGPipeline,
)


def test_local_hash_embedding_is_deterministic_and_normalized():
    provider = LocalHashEmbeddingProvider(dimensions=32)
    v1 = provider.embed_one("hello world")
    v2 = provider.embed_one("hello world")
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_vector_store_upsert_and_query_ranks_by_similarity():
    provider = LocalHashEmbeddingProvider(dimensions=32)
    store = InMemoryVectorStore()
    vectors = provider.embed(["cats are great pets", "stock market crashed today"])
    store.upsert(vectors, ["cats are great pets", "stock market crashed today"])

    query_vector = provider.embed_one("cats are great pets")
    results = store.query(query_vector, top_k=1)
    assert results[0][0].text == "cats are great pets"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_vector_store_ttl_expiry():
    store = InMemoryVectorStore()
    store.upsert([[1.0, 0.0]], ["expiring"], ttl_seconds=0.0)
    import time

    time.sleep(0.01)
    assert store.query([1.0, 0.0], top_k=5) == []


def test_semantic_chunk_respects_max_chars_and_overlap():
    text = "Sentence one. Sentence two. Sentence three. " * 20
    chunks = semantic_chunk(text, max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_semantic_chunk_empty_text():
    assert semantic_chunk("   ") == []


def test_rag_pipeline_ingest_and_retrieve():
    provider = LocalHashEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore()
    pipeline = RAGPipeline(provider, store, chunk_size=200, chunk_overlap=20)

    pipeline.ingest("Aegis is an AI platform with policy gates and evidence chains.")
    chunks = pipeline.retrieve("What is Aegis?", top_k=1)
    assert len(chunks) == 1
    assert "Aegis" in chunks[0].text

    prompt = pipeline.augment_prompt("What is Aegis?", top_k=1)
    assert "Aegis" in prompt
    assert "Question: What is Aegis?" in prompt

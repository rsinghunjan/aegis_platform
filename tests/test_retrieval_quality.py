import json
from types import SimpleNamespace

import pytest

from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow, AIWorkflowError
from services.embeddings import LocalHashEmbeddingProvider
from services.embeddings.rerank import LexicalReranker, LLMReranker
from services.grounding import check_grounding
from services.inference import InferenceProvider, InferenceResult, ModelRouter


def chunk(text, score):
    return SimpleNamespace(text=text, score=score, metadata={})


CHUNKS = [
    chunk("Weather today is sunny and warm.", 0.55),
    chunk("Evidence anchoring stores chain heads in a transparency log.", 0.5),
    chunk("Lunch menu includes soup.", 0.6),
]


def test_lexical_reranker_promotes_term_match():
    ranked = LexicalReranker().rerank("how does evidence anchoring work", CHUNKS, 2)
    assert "anchoring" in ranked[0].text and len(ranked) == 2


def test_llm_reranker_uses_scores_and_falls_back_on_garbage():
    llm = LLMReranker(lambda _p: json.dumps({"scores": [1, 2, 9]}))
    assert llm.rerank("q", CHUNKS, 1)[0].text.startswith("Lunch")
    for bad in ("nope", '{"scores": [1]}', '{"scores": [true, 1, 2]}'):
        got = LLMReranker(lambda _p, b=bad: b).rerank("evidence anchoring", CHUNKS, 1)
        assert "anchoring" in got[0].text


def test_grounding_flags_unsupported_sentences_and_accepts_abstention():
    context = "Aegis signs model artifacts before promotion."
    ok = check_grounding("Aegis signs model artifacts.", context)
    assert ok["grounded"] and ok["score"] == 1.0
    bad = check_grounding("Aegis signs model artifacts. It also mines bitcoin hourly.", context)
    assert not bad["grounded"] and bad["unsupported"] == ["It also mines bitcoin hourly."]
    assert check_grounding("I do not know.", context)["abstained"]


class Fixed(InferenceProvider):
    name = "fixed"

    def __init__(self, text):
        self.text = text

    def generate(self, request):
        return InferenceResult(self.text, "m", self.name, 1, 1, 1.0)


def workflow(tmp_path, text, monkeypatch, mode):
    monkeypatch.setenv("AEGIS_GROUNDING_MODE", mode)
    return AIWorkflow(
        inference_router=ModelRouter([Fixed(text)]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'g.db'}"),
    )


def test_enforce_mode_blocks_hallucinated_answer(tmp_path, monkeypatch):
    wf = workflow(tmp_path, "Aegis mines bitcoin on quantum GPUs.", monkeypatch, "enforce")
    wf.ingest("t", "Aegis signs model artifacts before promotion.")
    with pytest.raises(AIWorkflowError, match="not grounded"):
        wf.answer("t", "What does Aegis do?")
    assert wf.list_usage("t")[0]["status"] == "ungrounded"


def test_warn_mode_reports_grounding(tmp_path, monkeypatch):
    wf = workflow(tmp_path, "Aegis signs model artifacts.", monkeypatch, "warn")
    wf.ingest("t", "Aegis signs model artifacts before promotion.")
    assert wf.answer("t", "What does Aegis sign?")["grounding"]["grounded"] is True


def test_capacity_env_override_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_AI_MAX_CHUNKS_PER_TENANT", "7")
    assert workflow(tmp_path, "x", monkeypatch, "off").max_chunks_per_tenant == 7
    monkeypatch.setenv("AEGIS_AI_MAX_CHUNKS_PER_TENANT", "0")
    with pytest.raises(ValueError):
        workflow(tmp_path, "x", monkeypatch, "off")

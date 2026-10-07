"""Detalhes do Gemini testados com o SDK substituído por objetos falsos (sem rede, sem chave)."""

from types import SimpleNamespace

import numpy as np

from app.llm import embeddings
from app.llm.prompts import SYSTEM_PROMPT


def test_prefixos_de_embedding_do_lab4():
    assert embeddings.preparar_pergunta("o que é RAG?") == "task: question answering | query: o que é RAG?"
    assert embeddings.preparar_documento("Lab 4", "texto") == "title: Lab 4 | text: texto"


def test_embedding_usa_o_modelo_configurado(monkeypatch):
    chamadas = []

    def falso(**kwargs):
        chamadas.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.25])])

    monkeypatch.setattr(embeddings, "embutir_com_retry", falso)
    vetor = embeddings.embutir_pergunta("oi")
    assert np.allclose(vetor, [0.5, 0.25])
    assert chamadas[0]["contents"].startswith("task: question answering")
    assert chamadas[0]["model"] == "gemini-embedding-2"


def test_system_prompt_traz_as_regras_do_lab4():
    assert "apenas o CONTEXTO" in SYSTEM_PROMPT
    assert "Não encontrei essa informação no material da disciplina." in SYSTEM_PROMPT
    assert "[arquivo.md]" in SYSTEM_PROMPT

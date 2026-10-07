"""Adaptadores do Gemini testados com o SDK substituído por objetos falsos (sem rede, sem chave)."""

from types import SimpleNamespace

import numpy as np

from app.business.schemas import Trecho, TrechoRecuperado
from app.llm import assistente, embeddings
from app.llm.prompts import SYSTEM_PROMPT


def test_prefixos_de_embedding_do_lab4():
    assert embeddings.preparar_pergunta("o que é RAG?") == "task: question answering | query: o que é RAG?"
    assert embeddings.preparar_documento("Lab 4", "texto") == "title: Lab 4 | text: texto"


def test_embedder_usa_o_modelo_configurado(monkeypatch):
    chamadas = []

    def falso(**kwargs):
        chamadas.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5, 0.25])])

    monkeypatch.setattr(embeddings, "embutir_com_retry", falso)
    vetor = embeddings.GeminiEmbedder().embutir_pergunta("oi")
    assert np.allclose(vetor, [0.5, 0.25])
    assert chamadas[0]["contents"].startswith("task: question answering")
    assert chamadas[0]["model"] == "gemini-embedding-2"


def test_gerador_envia_contexto_system_prompt_e_encadeia_a_conversa(monkeypatch):
    chamadas = []

    def falso(**kwargs):
        chamadas.append(kwargs)
        return SimpleNamespace(output_text="Resposta [a.md]", id="int-7")

    monkeypatch.setattr(assistente, "interagir_com_retry", falso)
    trecho = Trecho(fonte="a.md", titulo="T", secao="T > S", url="u", conteudo="conteúdo do trecho")
    geracao = assistente.GeminiGerador().gerar("Pergunta?", [TrechoRecuperado(trecho=trecho, similaridade=0.9)], "int-6")

    assert geracao.texto == "Resposta [a.md]" and geracao.estado == "int-7"
    envio = chamadas[0]
    assert envio["previous_interaction_id"] == "int-6"
    assert envio["system_instruction"] == SYSTEM_PROMPT
    assert envio["input"].startswith("CONTEXTO:\nFONTE: [a.md]") and envio["input"].endswith("PERGUNTA:\nPergunta?")


def test_system_prompt_traz_as_regras_do_lab4():
    assert "apenas o CONTEXTO" in SYSTEM_PROMPT
    assert "Não encontrei essa informação no material da disciplina." in SYSTEM_PROMPT
    assert "[arquivo.md]" in SYSTEM_PROMPT

"""O fluxo do RAG (llm/assistente.py) com o Gemini trocado por fakes (fixture `ia`)."""

import numpy as np

from app import config
from app.business.schemas import ItemIndice, Trecho
from app.llm import assistente, embeddings
from tests.conftest import vetor_falso


def _indice():
    textos = {
        "aulas/lab4.md": ("Lab 4 > Similaridade de cosseno", "A similaridade de cosseno ordena os trechos por relevância"),
        "aulas/lab2.md": ("Lab 2 > Memória", "O previous_interaction_id mantém o contexto da conversa"),
    }
    itens = []
    for fonte, (secao, conteudo) in textos.items():
        t = Trecho(fonte=fonte, titulo=secao.split(" > ")[0], secao=secao, url=f"http://x/{fonte}", conteudo=conteudo)
        itens.append(ItemIndice(t, vetor_falso(f"{secao} {conteudo}")))
    return itens


def test_conversar_cita_fonte_guarda_estado_e_envia_o_anterior(ia):
    resposta = assistente.conversar("O que é similaridade de cosseno?", [], _indice(), estado_anterior="e0")
    assert resposta.fundamentada
    assert resposta.fontes[0].fonte == "aulas/lab4.md" and resposta.fontes[0].citada
    assert resposta.estado == "estado-1"
    assert ia.chamadas[0]["previous_interaction_id"] == "e0"      # a memória da conversa chegou ao modelo
    assert resposta.citacoes_invalidas == []


def test_sem_contexto_suficiente_nao_chama_o_modelo(ia, monkeypatch):
    monkeypatch.setattr(config, "SIMILARIDADE_MINIMA", 0.5)
    resposta = assistente.conversar("receita de bolo de cenoura", [], _indice(), estado_anterior="e0")
    assert not resposta.fundamentada
    assert resposta.texto.startswith("Não encontrei")
    assert resposta.fontes == []
    assert resposta.estado == "e0"               # a conversa não "avança" sem geração
    assert ia.chamadas == []


def test_envia_contexto_system_prompt_e_modelo_configurado(ia, monkeypatch):
    monkeypatch.setattr(config, "SIMILARIDADE_MINIMA", 0.15)
    assistente.conversar("Para que serve o previous_interaction_id?", [], _indice())
    envio = ia.chamadas[0]
    assert envio["model"] == config.GEMINI_MODEL
    assert "apenas o CONTEXTO" in envio["system_instruction"]
    assert envio["input"].startswith("CONTEXTO:\nFONTE: [aulas/lab2.md]")
    assert envio["input"].endswith("PERGUNTA:\nPara que serve o previous_interaction_id?")


def test_citacao_inventada_e_sinalizada(ia):
    ia.texto = "Algo [inventada.md]"
    resposta = assistente.conversar("similaridade de cosseno", [], _indice())
    assert resposta.citacoes_invalidas == ["inventada.md"]
    assert not any(f.citada for f in resposta.fontes)


def test_pergunta_curta_usa_historico_na_busca(ia):
    resposta = assistente.conversar("e o cosseno?", ["Explique similaridade de trechos"], _indice())
    assert resposta.consulta == "Explique similaridade de trechos e o cosseno?"


def test_busca_funciona_com_vetores_escritos_a_mao(monkeypatch):
    """A recuperação só precisa de um vetor da pergunta e do índice: não depende de qual modelo gerou os vetores."""
    monkeypatch.setattr(embeddings, "embutir_pergunta", lambda pergunta: np.array([0.9, 0.1]))
    trecho = Trecho(fonte="x.md", titulo="X", secao="X", url="u", conteudo="conteúdo de teste")
    _, trechos = assistente.recuperar("qualquer coisa", [], [ItemIndice(trecho, np.array([1.0, 0.0]))])
    assert trechos[0].trecho.fonte == "x.md" and trechos[0].similaridade > 0.99

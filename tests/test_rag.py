"""O caso de uso RAG roda igual com qualquer Embedder/Gerador — inclusive sem IA."""

import numpy as np

from app.business import rag
from app.business.extrativo import GeradorExtrativo
from app.business.schemas import ItemIndice, Trecho
from tests.conftest import EmbedderFalso, GeradorFalso


def _indice(embedder):
    textos = {
        "aulas/lab4.md": ("Lab 4 > Similaridade de cosseno", "A similaridade de cosseno ordena os trechos por relevância"),
        "aulas/lab2.md": ("Lab 2 > Memória", "O previous_interaction_id mantém o contexto da conversa"),
    }
    itens = []
    for fonte, (secao, conteudo) in textos.items():
        t = Trecho(fonte=fonte, titulo=secao.split(" > ")[0], secao=secao, url=f"http://x/{fonte}", conteudo=conteudo)
        itens.append(ItemIndice(t, embedder.embutir_documento(secao, conteudo)))
    return itens


def test_responder_com_gerador_falso_cita_fonte_e_guarda_estado():
    emb, ger = EmbedderFalso(), GeradorFalso()
    resposta = rag.responder("O que é similaridade de cosseno?", [], _indice(emb), emb, ger, estado_anterior="e0")
    assert resposta.fundamentada
    assert resposta.fontes[0].fonte == "aulas/lab4.md" and resposta.fontes[0].citada
    assert resposta.estado == "estado-1"
    assert ger.chamadas[0][1] == "e0"          # o estado anterior chegou ao gerador
    assert resposta.citacoes_invalidas == []


def test_sem_contexto_suficiente_nao_chama_o_gerador():
    emb, ger = EmbedderFalso(), GeradorFalso()
    resposta = rag.responder("receita de bolo de cenoura", [], _indice(emb), emb, ger,
                             estado_anterior="e0", similaridade_minima=0.5)
    assert not resposta.fundamentada
    assert resposta.texto.startswith("Não encontrei")
    assert resposta.fontes == []
    assert resposta.estado == "e0"             # a conversa não é "avançada" sem geração
    assert ger.chamadas == []


def test_funciona_sem_nenhuma_ia_gerador_extrativo():
    emb = EmbedderFalso()
    resposta = rag.responder("Para que serve o previous_interaction_id?", [], _indice(emb), emb, GeradorExtrativo())
    assert "mantém o contexto da conversa" in resposta.texto
    assert resposta.fontes[0].fonte == "aulas/lab2.md" and resposta.fontes[0].citada


def test_funciona_com_vetores_escritos_a_mao_sem_embedder_real():
    class EmbedderManual:
        def embutir_documento(self, titulo, conteudo):
            return np.array([1.0, 0.0])

        def embutir_pergunta(self, pergunta):
            return np.array([0.9, 0.1])

    trecho = Trecho(fonte="x.md", titulo="X", secao="X", url="u", conteudo="conteúdo de teste")
    resposta = rag.responder("qualquer coisa", [], [ItemIndice(trecho, np.array([1.0, 0.0]))],
                             EmbedderManual(), GeradorExtrativo())
    assert resposta.fundamentada and resposta.fontes[0].fonte == "x.md"


def test_citacao_inventada_e_sinalizada():
    class GeradorQueInventa:
        def gerar(self, pergunta, trechos, estado):
            from app.business.schemas import Geracao
            return Geracao(texto="Algo [inventada.md]", estado=None)

    emb = EmbedderFalso()
    resposta = rag.responder("similaridade de cosseno", [], _indice(emb), emb, GeradorQueInventa())
    assert resposta.citacoes_invalidas == ["inventada.md"]
    assert not any(f.citada for f in resposta.fontes)


def test_pergunta_curta_usa_historico_na_busca():
    emb = EmbedderFalso()
    resposta = rag.responder("e o cosseno?", ["Explique similaridade de trechos"], _indice(emb), emb, GeradorFalso())
    assert resposta.consulta == "Explique similaridade de trechos e o cosseno?"

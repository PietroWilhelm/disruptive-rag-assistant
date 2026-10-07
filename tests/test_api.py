"""Testes da API com o Gemini trocado por fakes (fixture `ia`) — nenhum teste usa a rede nem GEMINI_API_KEY."""

import pytest
from fastapi.testclient import TestClient

from app import config, ingestao
from app.llm.client import CotaEsgotadaError
from app.main import app


@pytest.fixture()
def cliente(ia, banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    monkeypatch.setattr(config, "AUTO_INGESTAO", False)
    monkeypatch.setattr(config, "SIMILARIDADE_MINIMA", 0.15)
    ingestao.executar(progresso=lambda *_: None)
    with TestClient(app) as c:
        c.ia = ia
        yield c


def _sessao(c):
    return c.post("/api/sessoes").json()["sessao_id"]


def test_health_mostra_base_indexada(cliente):
    corpo = cliente.get("/api/health").json()
    assert corpo["status"] == "ok" and corpo["base_indexada"] and corpo["trechos"] > 3


def test_frontend_e_servido_na_raiz(cliente):
    resposta = cliente.get("/")
    assert resposta.status_code == 200 and "Assistente" in resposta.text


def test_criar_sessao_e_404(cliente):
    assert cliente.post("/api/sessoes").status_code == 201
    assert cliente.get("/api/sessoes/nao-existe").status_code == 404
    assert cliente.post("/api/sessoes/nao-existe/mensagens", json={"mensagem": "oi tudo bem"}).status_code == 404


def test_pergunta_com_fontes_e_historico(cliente):
    sessao = _sessao(cliente)
    r = cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "O que é similaridade de cosseno?"})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["fundamentada"] and corpo["resposta"].startswith("Resposta de teste.")
    assert corpo["fontes"][0]["fonte"] == "aulas/genAI/lab4/lab4.md" and corpo["fontes"][0]["citada"]
    assert corpo["fontes"][0]["url"].endswith("/aulas/genAI/lab4/lab4/#similaridade-de-cosseno")

    historico = cliente.get(f"/api/sessoes/{sessao}").json()
    assert [m["remetente"] for m in historico["mensagens"]] == ["aluno", "assistente"]
    assert historico["mensagens"][1]["fontes"][0]["citada"] is True


def test_memoria_da_conversa_encadeia_o_estado(cliente):
    sessao = _sessao(cliente)
    cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "O que é similaridade de cosseno?"})
    cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "E os embeddings?"})
    estados_recebidos = [c["previous_interaction_id"] for c in cliente.ia.chamadas]
    assert estados_recebidos == [None, "estado-1"]


def test_fora_de_escopo_responde_sem_gerar(cliente, monkeypatch):
    monkeypatch.setattr(config, "SIMILARIDADE_MINIMA", 0.95)
    sessao = _sessao(cliente)
    corpo = cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "receita de bolo de cenoura"}).json()
    assert corpo["fundamentada"] is False and corpo["fontes"] == []
    assert corpo["resposta"] == "Não encontrei essa informação no material da disciplina."
    assert cliente.ia.chamadas == []


def test_busca_sem_geracao(cliente):
    corpo = cliente.post("/api/buscar", json={"pergunta": "previous_interaction_id", "k": 2}).json()
    assert len(corpo["trechos"]) == 2
    assert corpo["trechos"][0]["fonte"] == "aulas/genAI/lab2/lab2.md"
    assert cliente.ia.chamadas == []


def test_validacao_da_mensagem(cliente):
    sessao = _sessao(cliente)
    assert cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "x"}).status_code == 422
    assert cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "a" * 801}).status_code == 422


def test_limite_de_mensagens_por_minuto(cliente, monkeypatch):
    monkeypatch.setattr(config, "LIMITE_MENSAGENS_POR_MINUTO", 2)
    sessao = _sessao(cliente)
    codigos = [
        cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "similaridade de cosseno"}).status_code
        for _ in range(3)
    ]
    assert codigos == [200, 200, 429]


def test_503_quando_a_base_esta_vazia(ia, banco, monkeypatch):
    monkeypatch.setattr(config, "AUTO_INGESTAO", False)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["base_indexada"] is False
        sessao = c.post("/api/sessoes").json()["sessao_id"]
        assert c.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "similaridade de cosseno"}).status_code == 503


def test_cota_esgotada_vira_429_amigavel(cliente):
    cliente.ia.erro = CotaEsgotadaError("A cota DIARIA da API do Gemini acabou")
    sessao = _sessao(cliente)
    r = cliente.post(f"/api/sessoes/{sessao}/mensagens", json={"mensagem": "similaridade de cosseno"})
    assert r.status_code == 429 and "cota" in r.json()["detail"].lower()


def test_banco_vazio_usa_o_indice_pronto_sem_chamar_a_api(ia, banco, material, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    ingestao.executar(progresso=lambda *_: None)
    semente = tmp_path / "seed" / "indice.db"
    total = banco.exportar_indice(str(semente))

    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "deploy-novo.db"))    # banco vazio
    monkeypatch.setattr(config, "SEMENTE_PATH", str(semente))
    monkeypatch.setattr(config, "AUTO_INGESTAO", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "chave-falsa")
    docs_antes = ia.chamadas_documento
    with TestClient(app) as c:
        corpo = c.get("/api/health").json()
    assert corpo["trechos"] == total and corpo["base_indexada"] and not corpo["indexando"]
    assert ia.chamadas_documento == docs_antes                  # não reindexou nada

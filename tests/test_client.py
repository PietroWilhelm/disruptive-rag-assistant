"""Retry, cota e ritmo do cliente do Gemini — com erros falsos e sem dormir de verdade."""

import pytest
from google.genai import errors as genai_errors

from app.llm import client
from app.llm.client import CotaEsgotadaError


def _erro_429(quota_id: str, retry="46s"):
    corpo = {"error": {"code": 429, "message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED",
                       "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [{"quotaId": quota_id}]},
                                   {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}]}}
    return genai_errors.ClientError(429, corpo)


@pytest.fixture()
def sem_dormir(monkeypatch):
    esperas = []
    monkeypatch.setattr(client.time, "sleep", lambda s: esperas.append(s))
    return esperas


def test_distingue_cota_diaria_de_cota_por_minuto():
    assert client.cota_diaria_esgotada(_erro_429("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not client.cota_diaria_esgotada(_erro_429("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier"))


def test_429_por_minuto_espera_o_tempo_sugerido_e_tenta_de_novo(sem_dormir):
    tentativas = []

    def funcao(**kwargs):
        tentativas.append(1)
        if len(tentativas) < 3:
            raise _erro_429("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier")
        return "ok"

    assert client._com_retry(funcao) == "ok"
    assert len(tentativas) == 3
    assert sem_dormir[0] == 47.0          # 46s sugeridos pela API + 1s de margem


def test_cota_diaria_nao_adianta_esperar(sem_dormir):
    def funcao(**kwargs):
        raise _erro_429("EmbedContentRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(CotaEsgotadaError, match="DIARIA"):
        client._com_retry(funcao)
    assert sem_dormir == []               # nem tentou esperar


def test_429_que_nao_passa_vira_cota_esgotada(sem_dormir):
    def funcao(**kwargs):
        raise _erro_429("EmbedContentRequestsPerMinutePerProjectPerModel-FreeTier")

    with pytest.raises(CotaEsgotadaError):
        client._com_retry(funcao, max_tentativas=3)
    assert len(sem_dormir) == 2


def test_outros_erros_nao_sao_repetidos(sem_dormir):
    def funcao(**kwargs):
        raise genai_errors.ClientError(400, {"error": {"code": 400, "message": "ruim", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(genai_errors.ClientError):
        client._com_retry(funcao)
    assert sem_dormir == []


def test_ritmo_limita_chamadas_por_minuto(monkeypatch):
    relogio = {"agora": 100.0}
    esperas = []
    monkeypatch.setattr(client.time, "monotonic", lambda: relogio["agora"])
    monkeypatch.setattr(client.time, "sleep", lambda s: (esperas.append(s), relogio.__setitem__("agora", relogio["agora"] + s)))
    client._ritmo["ultima_chamada"] = 0.0

    for _ in range(4):
        client.respeitar_ritmo(60)        # 60 por minuto = 1 por segundo
    assert esperas == pytest.approx([1.0, 1.0, 1.0])    # a 1ª passa direto, as outras esperam 1s


def test_ritmo_zero_desliga_o_controle(monkeypatch):
    chamadas = []
    monkeypatch.setattr(client.time, "sleep", lambda s: chamadas.append(s))
    client.respeitar_ritmo(0)
    assert chamadas == []

"""Fábricas das portas usadas pelas rotas (injeção de dependência do FastAPI).

É aqui, e só aqui, que a API escolhe QUEM implementa `Embedder` e `Gerador`.
Nos testes elas são trocadas por fakes com `app.dependency_overrides`.
"""

from app import config
from app.business.extrativo import GeradorExtrativo
from app.business.portas import Embedder, Gerador


def obter_embedder() -> Embedder:
    from app.llm.embeddings import GeminiEmbedder

    return GeminiEmbedder()


def obter_gerador() -> Gerador:
    if config.MODO_GERACAO == "extrativo":
        return GeradorExtrativo()
    from app.llm.assistente import GeminiGerador

    return GeminiGerador()

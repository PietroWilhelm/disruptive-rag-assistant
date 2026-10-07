"""Fakes compartilhados pelos testes. Nenhum teste usa a internet ou GEMINI_API_KEY."""

import hashlib
import json
import unicodedata
import re

import numpy as np
import pytest

from app.business.schemas import Geracao, TrechoRecuperado
from app.api import limite

DIMENSAO = 128

LAB4_MD = """---
title: Lab 4
---
# Lab 4 - RAG e bases de conhecimento

Neste laboratório você aprende a recuperar informação antes de gerar a resposta.

## Similaridade de cosseno

A similaridade de cosseno ordena os trechos por relevância, de -1 a 1.

```python
def similaridade_cosseno(a, b):
    # este # nao e um titulo
    return a @ b
```

## Embeddings

Embeddings são coordenadas em um espaço vetorial: textos parecidos ficam próximos.
"""

LAB2_MD = """# Lab 2 - Assistente Conversacional

## Memória da conversa

O previous_interaction_id encadeia as interações e mantém o contexto da conversa.
"""

CHECKPOINT_IPYNB = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Checkpoint 4 - Pet Shop\n", "\n", "Assistente de banho, tosa e hidratação."]},
        {"cell_type": "code", "source": ["from pydantic import BaseModel\n", "print('ficha')"], "outputs": [{"text": "SAIDA IGNORADA"}]},
        {"cell_type": "markdown", "source": ["## Guardrail veterinário\n", "\n", "Se o pet estiver doente, recuse o agendamento e oriente procurar um veterinário."]},
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _palavras(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [p for p in re.findall(r"[a-z0-9_]{3,}", texto)]


def vetor_falso(texto: str) -> np.ndarray:
    """Embedding determinístico por hash de palavras: textos com palavras em comum ficam próximos."""
    v = np.zeros(DIMENSAO, dtype=float)
    for p in _palavras(texto):
        v[int(hashlib.md5(p.encode()).hexdigest(), 16) % DIMENSAO] += 1.0
    return v


class EmbedderFalso:
    def __init__(self):
        self.chamadas_documento = 0

    def embutir_documento(self, titulo, conteudo):
        self.chamadas_documento += 1
        return vetor_falso(f"{titulo} {conteudo}")

    def embutir_pergunta(self, pergunta):
        return vetor_falso(pergunta)


class GeradorFalso:
    """Cita a fonte do primeiro trecho, como o modelo deveria fazer."""

    def __init__(self):
        self.chamadas = []

    def gerar(self, pergunta, trechos: list[TrechoRecuperado], estado):
        self.chamadas.append((pergunta, estado))
        fonte = trechos[0].trecho.fonte
        return Geracao(texto=f"Resposta de teste. [{fonte}]", estado=f"estado-{len(self.chamadas)}")


@pytest.fixture()
def material(tmp_path):
    pasta = tmp_path / "material"
    (pasta / "aulas" / "genAI" / "lab4").mkdir(parents=True)
    (pasta / "aulas" / "genAI" / "lab2").mkdir(parents=True)
    (pasta / "aulas" / "checkpoint").mkdir(parents=True)
    (pasta / "aulas" / "genAI" / "lab4" / "lab4.md").write_text(LAB4_MD, encoding="utf-8")
    (pasta / "aulas" / "genAI" / "lab2" / "lab2.md").write_text(LAB2_MD, encoding="utf-8")
    (pasta / "aulas" / "checkpoint" / "cp4.ipynb").write_text(json.dumps(CHECKPOINT_IPYNB), encoding="utf-8")
    (pasta / "imagem.png").write_bytes(b"\x89PNG")  # ignorado
    return pasta


@pytest.fixture()
def banco(tmp_path, monkeypatch):
    from app import config
    from app.persistence import db

    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "teste.db"))
    db.inicializar_schema()
    return db


@pytest.fixture(autouse=True)
def _zerar_limite(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "LIMITE_MENSAGENS_POR_MINUTO", 1000)
    limite.limpar()

"""Portas (contratos) que a lógica de negócio exige do mundo externo.

A camada de negócio não sabe QUEM gera embeddings nem QUEM escreve a resposta:
ela só define o que precisa receber. Hoje `app/llm/` implementa estas portas
com o Gemini; amanhã poderia ser outro modelo — ou nenhum, como o
`GeradorExtrativo` (app/business/extrativo.py), que responde sem IA.
"""

from typing import Protocol

import numpy as np

from app.business.schemas import Geracao, TrechoRecuperado


class Embedder(Protocol):
    def embutir_documento(self, titulo: str, conteudo: str) -> np.ndarray:
        """Vetor de um trecho do material."""
        ...

    def embutir_pergunta(self, pergunta: str) -> np.ndarray:
        """Vetor de uma pergunta (precisa estar no mesmo espaço dos documentos)."""
        ...


class Gerador(Protocol):
    def gerar(self, pergunta: str, trechos: list[TrechoRecuperado], estado: str | None) -> Geracao:
        """Escreve a resposta usando SOMENTE os trechos recebidos."""
        ...

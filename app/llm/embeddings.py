"""Embeddings com o Gemini (Lab 4).

O prefixo do texto muda conforme o papel (pergunta x documento): é uma exigência
do modelo `gemini-embedding-2`, por isso fica aqui e não na camada de negócio.
"""

import numpy as np

from app import config
from app.llm.client import embutir_com_retry


def preparar_pergunta(pergunta: str) -> str:
    return f"task: question answering | query: {pergunta}"


def preparar_documento(titulo: str, conteudo: str) -> str:
    return f"title: {titulo} | text: {conteudo}"


def gerar_embedding(texto: str) -> np.ndarray:
    resultado = embutir_com_retry(model=config.EMBEDDING_MODEL, contents=texto)
    return np.array(resultado.embeddings[0].values, dtype=float)


def embutir_pergunta(pergunta: str) -> np.ndarray:
    return gerar_embedding(preparar_pergunta(pergunta))


def embutir_documento(titulo: str, conteudo: str) -> np.ndarray:
    return gerar_embedding(preparar_documento(titulo, conteudo))

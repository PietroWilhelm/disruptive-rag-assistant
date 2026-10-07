"""Busca por similaridade (Lab 4): cosseno entre o vetor da pergunta e os vetores dos trechos.

Recebe VETORES já prontos — quem os gera (qual modelo) não é problema desta camada.
"""

import numpy as np

from app.business.schemas import ItemIndice, TrechoRecuperado


def similaridade_cosseno(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    a = np.asarray(vetor_a, dtype=float)
    b = np.asarray(vetor_b, dtype=float)
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    if denominador == 0:
        return 0.0
    return float(np.dot(a, b) / denominador)


def buscar_trechos(vetor_pergunta: np.ndarray, indice: list[ItemIndice], k: int = 4) -> list[TrechoRecuperado]:
    """Top-k trechos mais parecidos com a pergunta, do mais para o menos similar."""
    resultados = [
        TrechoRecuperado(trecho=item.trecho, similaridade=similaridade_cosseno(vetor_pergunta, item.embedding))
        for item in indice
    ]
    resultados.sort(key=lambda r: r.similaridade, reverse=True)
    return resultados[: max(k, 0)]

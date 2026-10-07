"""Caso de uso RAG: pergunta -> recuperação -> (contexto suficiente?) -> geração -> fontes.

Orquestra só funções de negócio e as duas portas (`Embedder`, `Gerador`). Não importa
nenhum SDK de IA, banco ou framework web.
"""

from app.business import regras
from app.business.busca import buscar_trechos
from app.business.portas import Embedder, Gerador
from app.business.schemas import ItemIndice, RespostaRag, TrechoRecuperado


def recuperar(
    pergunta: str,
    perguntas_anteriores: list[str],
    indice: list[ItemIndice],
    embedder: Embedder,
    k: int = 4,
) -> tuple[str, list[TrechoRecuperado]]:
    """Só a etapa de recuperação (o Lab 4 pede para avaliá-la separada da geração)."""
    consulta = regras.montar_consulta(pergunta, perguntas_anteriores)
    vetor = embedder.embutir_pergunta(consulta)
    return consulta, buscar_trechos(vetor, indice, k)


def responder(
    pergunta: str,
    perguntas_anteriores: list[str],
    indice: list[ItemIndice],
    embedder: Embedder,
    gerador: Gerador,
    estado_anterior: str | None = None,
    k: int = 4,
    similaridade_minima: float = 0.0,
) -> RespostaRag:
    consulta, trechos = recuperar(pergunta, perguntas_anteriores, indice, embedder, k)

    if not regras.tem_contexto_suficiente(trechos, similaridade_minima):
        return RespostaRag(
            texto=regras.MENSAGEM_SEM_RESPOSTA,
            fontes=[],
            fundamentada=False,
            estado=estado_anterior,
            consulta=consulta,
        )

    geracao = gerador.gerar(pergunta, trechos, estado_anterior)
    validas, invalidas = regras.classificar_citacoes(geracao.texto, trechos)
    return RespostaRag(
        texto=geracao.texto,
        fontes=regras.montar_fontes(trechos, validas),
        fundamentada=True,
        estado=geracao.estado,
        consulta=consulta,
        citacoes_invalidas=invalidas,
    )

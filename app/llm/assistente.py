"""Orquestra o RAG com o Gemini: pergunta -> busca -> contexto suficiente? -> resposta -> fontes.

É a camada que fala com a IA (embeddings e Interactions API). As decisões em si
(montar a consulta, o limiar de similaridade, o contexto, a conferência das
citações) vêm de `app.business`, que não sabe nada de Gemini: aqui só se chama
o modelo no meio do caminho.

A memória da conversa vem do encadeamento de `previous_interaction_id` (Lab 2):
o `id` da última interação volta em `RespostaRag.estado`, é guardado por sessão
em `app.persistence` e entregue de novo na próxima mensagem. A base de
conhecimento (Lab 4) chega a cada mensagem dentro do `input`, como CONTEXTO.
"""

from app import config
from app.business import regras
from app.business.busca import buscar_trechos
from app.business.schemas import ItemIndice, RespostaRag, TrechoRecuperado
from app.llm import embeddings
from app.llm.client import interagir_com_retry
from app.llm.prompts import SYSTEM_PROMPT


def recuperar(
    pergunta: str,
    perguntas_anteriores: list[str],
    indice: list[ItemIndice],
    k: int | None = None,
) -> tuple[str, list[TrechoRecuperado]]:
    """Só a recuperação, sem gerar resposta (o Lab 4 pede para avaliá-la separada da geração)."""
    consulta = regras.montar_consulta(pergunta, perguntas_anteriores)
    vetor = embeddings.embutir_pergunta(consulta)
    return consulta, buscar_trechos(vetor, indice, k or config.TOP_K)


def conversar(
    pergunta: str,
    perguntas_anteriores: list[str],
    indice: list[ItemIndice],
    estado_anterior: str | None = None,
) -> RespostaRag:
    consulta, trechos = recuperar(pergunta, perguntas_anteriores, indice)

    if not regras.tem_contexto_suficiente(trechos, config.SIMILARIDADE_MINIMA):
        return RespostaRag(
            texto=regras.MENSAGEM_SEM_RESPOSTA,
            fontes=[],
            fundamentada=False,
            estado=estado_anterior,
            consulta=consulta,
        )

    interaction = interagir_com_retry(
        model=config.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        input=f"CONTEXTO:\n{regras.montar_contexto(trechos)}\n\nPERGUNTA:\n{pergunta}",
        previous_interaction_id=estado_anterior,
    )
    texto = interaction.output_text or ""
    validas, invalidas = regras.classificar_citacoes(texto, trechos)
    return RespostaRag(
        texto=texto,
        fontes=regras.montar_fontes(trechos, validas),
        fundamentada=True,
        estado=interaction.id,
        consulta=consulta,
        citacoes_invalidas=invalidas,
    )

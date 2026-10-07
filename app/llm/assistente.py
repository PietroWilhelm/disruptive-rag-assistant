"""Gerador de respostas com o Gemini via Interactions API: implementa a porta `Gerador`.

A memória da conversa vem do encadeamento de `previous_interaction_id` (Lab 2): o
`interaction_id` da última resposta volta como `Geracao.estado`, é guardado por
sessão pela camada `app.persistence` e entregue de novo na próxima mensagem.
Já a base de conhecimento (Lab 4) chega a cada mensagem dentro do `input`, como
CONTEXTO, montado pela camada de negócio.
"""

from app import config
from app.business.regras import montar_contexto
from app.business.schemas import Geracao, TrechoRecuperado
from app.llm.client import interagir_com_retry
from app.llm.prompts import SYSTEM_PROMPT


class GeminiGerador:
    def gerar(self, pergunta: str, trechos: list[TrechoRecuperado], estado: str | None) -> Geracao:
        interaction = interagir_com_retry(
            model=config.GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            input=f"CONTEXTO:\n{montar_contexto(trechos)}\n\nPERGUNTA:\n{pergunta}",
            previous_interaction_id=estado,
        )
        return Geracao(texto=interaction.output_text or "", estado=interaction.id)

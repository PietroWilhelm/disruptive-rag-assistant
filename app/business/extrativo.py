"""Gerador sem IA: responde com o trecho mais similar, sem escrever nada novo.

Existe para provar (e testar) que a lógica de negócio funciona igual com qualquer
gerador — inclusive nenhum. Ative com MODO_GERACAO=extrativo.
"""

from app.business.schemas import Geracao, TrechoRecuperado


class GeradorExtrativo:
    def gerar(self, pergunta: str, trechos: list[TrechoRecuperado], estado: str | None = None) -> Geracao:
        melhor = trechos[0].trecho
        texto = f"Encontrei isto no material ({melhor.secao}):\n\n{melhor.conteudo}\n\n[{melhor.fonte}]"
        return Geracao(texto=texto, estado=estado)

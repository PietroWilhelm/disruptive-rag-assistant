"""Regras do assistente da disciplina — funções puras, sem IA.

São as decisões que NÃO podem depender do modelo generativo: quando há contexto
suficiente para responder, como o contexto é montado, como as fontes citadas
são conferidas. Dá para testar tudo isto sem chave de API (tests/test_regras.py).
"""

import re

from app.business.schemas import FonteCitada, TrechoRecuperado

MENSAGEM_SEM_RESPOSTA = "Não encontrei essa informação no material da disciplina."

# Perguntas curtas ("e no ESP32?") são completadas com a pergunta anterior do aluno
# antes da busca, porque o Lab 4 separa memória da conversa de base de conhecimento:
# a busca só enxerga a pergunta que recebe.
PALAVRAS_PERGUNTA_CURTA = 5

_RE_CITACAO = re.compile(r"\[([^\[\]\n]+?\.(?:md|ipynb))\]")


def montar_consulta(pergunta: str, perguntas_anteriores: list[str]) -> str:
    pergunta = pergunta.strip()
    if perguntas_anteriores and len(pergunta.split()) <= PALAVRAS_PERGUNTA_CURTA:
        return f"{perguntas_anteriores[-1].strip()} {pergunta}"
    return pergunta


def tem_contexto_suficiente(trechos: list[TrechoRecuperado], similaridade_minima: float) -> bool:
    """O cosseno ordena por relevância, mas não garante que o trecho responde (Lab 4).
    Abaixo do limiar nem chamamos o gerador: devolvemos MENSAGEM_SEM_RESPOSTA."""
    return bool(trechos) and trechos[0].similaridade >= similaridade_minima


def montar_contexto(trechos: list[TrechoRecuperado]) -> str:
    blocos = []
    for item in trechos:
        t = item.trecho
        blocos.append(
            f"FONTE: [{t.fonte}]\n"
            f"TÍTULO: {t.secao}\n"
            f"CONTEÚDO: {t.conteudo}"
        )
    return "\n\n---\n\n".join(blocos)


def extrair_citacoes(texto: str) -> list[str]:
    """Fontes citadas no formato [arquivo.md], sem repetição e na ordem em que aparecem."""
    vistas: list[str] = []
    for fonte in _RE_CITACAO.findall(texto):
        fonte = fonte.strip()
        if fonte not in vistas:
            vistas.append(fonte)
    return vistas


def classificar_citacoes(texto: str, trechos: list[TrechoRecuperado]) -> tuple[list[str], list[str]]:
    """Separa as citações em (válidas, inválidas): válida é a que aponta para um trecho recuperado."""
    conhecidas = {item.trecho.fonte for item in trechos}
    validas, invalidas = [], []
    for fonte in extrair_citacoes(texto):
        (validas if fonte in conhecidas else invalidas).append(fonte)
    return validas, invalidas


def montar_fontes(trechos: list[TrechoRecuperado], citacoes_validas: list[str]) -> list[FonteCitada]:
    """Uma entrada por seção recuperada, marcando as que o texto da resposta citou."""
    fontes = []
    vistos = set()
    for item in trechos:
        t = item.trecho
        chave = (t.fonte, t.secao)
        if chave in vistos:
            continue
        vistos.add(chave)
        fontes.append(
            FonteCitada(
                fonte=t.fonte,
                titulo=t.titulo,
                secao=t.secao,
                url=t.url,
                similaridade=round(item.similaridade, 4),
                citada=t.fonte in citacoes_validas,
            )
        )
    return fontes

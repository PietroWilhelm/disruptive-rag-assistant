"""Contratos de dados da camada de negócio.

Nada aqui conhece IA, banco de dados ou HTTP: são só estruturas de dados
(Pydantic, como no Lab 3, e um dataclass para o item do índice, que carrega
um vetor numpy como no Lab 4).
"""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Trecho(BaseModel):
    """Um pedaço do material da disciplina (uma seção, ou parte dela)."""

    model_config = ConfigDict(frozen=True)

    fonte: str = Field(description="Caminho do arquivo dentro do material, ex.: aulas/genAI/lab4/lab4.md")
    titulo: str = Field(description="Título da página")
    secao: str = Field(description="Caminho de títulos, ex.: Lab 4 > Arquitetura Conceitual")
    url: str = Field(description="Link da seção no site da disciplina")
    conteudo: str


@dataclass
class ItemIndice:
    """Um trecho já transformado em vetor (embedding)."""

    trecho: Trecho
    embedding: np.ndarray


class TrechoRecuperado(BaseModel):
    trecho: Trecho
    similaridade: float


class Geracao(BaseModel):
    """Resultado de um gerador de respostas (qualquer um: Gemini, outro LLM ou nenhum).

    `estado` é um valor opaco que o gerador devolve para continuar a conversa
    (no Gemini é o `interaction_id`). A camada de negócio nunca interpreta esse valor.
    """

    texto: str
    estado: str | None = None


class FonteCitada(BaseModel):
    fonte: str
    titulo: str
    secao: str
    url: str
    similaridade: float
    citada: bool = Field(description="True se o texto da resposta cita esta fonte com [arquivo.md]")


class RespostaRag(BaseModel):
    texto: str
    fontes: list[FonteCitada]
    fundamentada: bool = Field(description="False quando a base não tinha contexto suficiente e nenhuma geração foi feita")
    estado: str | None = None
    consulta: str = ""
    citacoes_invalidas: list[str] = Field(
        default_factory=list,
        description="Fontes citadas no texto que não estavam entre os trechos recuperados",
    )

"""Modelos de request/response da API (camada HTTP).

Ficam separados dos schemas de negócio (`app.business.schemas`) porque tratam do
formato da requisição/resposta HTTP, não da regra de negócio em si.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    base_indexada: bool
    trechos: int
    indexando: bool
    modo_geracao: str
    erro_ingestao: str = ""


class NovaSessaoResponse(BaseModel):
    sessao_id: str


class NovaMensagemRequest(BaseModel):
    mensagem: str = Field(min_length=2, max_length=800, description="Pergunta do aluno.")


class FonteResponse(BaseModel):
    fonte: str
    titulo: str
    secao: str
    url: str
    similaridade: float
    citada: bool


class NovaMensagemResponse(BaseModel):
    resposta: str
    fundamentada: bool
    fontes: list[FonteResponse]


class BuscaRequest(BaseModel):
    pergunta: str = Field(min_length=2, max_length=800)
    k: int = Field(default=4, ge=1, le=20)


class TrechoBuscaResponse(BaseModel):
    fonte: str
    secao: str
    url: str
    similaridade: float
    conteudo: str


class BuscaResponse(BaseModel):
    consulta: str
    trechos: list[TrechoBuscaResponse]


class HistoricoResponse(BaseModel):
    sessao_id: str
    mensagens: list[dict]


class ReindexacaoResponse(BaseModel):
    status: str

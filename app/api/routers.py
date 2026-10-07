"""Rotas HTTP do assistente da disciplina.

Esta é a única camada que conhece HTTP (FastAPI). Ela só orquestra: recebe a
requisição, chama o caso de uso `app.business.rag` (com as portas escolhidas em
`dependencias`), grava em `app.persistence.db` e devolve a resposta.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app import config
from app.api import dependencias, indexacao
from app.api.limite import verificar_limite
from app.api.schemas import (
    BuscaRequest,
    BuscaResponse,
    FonteResponse,
    HealthResponse,
    HistoricoResponse,
    NovaMensagemRequest,
    NovaMensagemResponse,
    NovaSessaoResponse,
    ReindexacaoResponse,
    TrechoBuscaResponse,
)
from app.business import rag
from app.business.portas import Embedder, Gerador
from app.llm.client import ERROS_DA_API, ChaveApiAusenteError, descrever_erro
from app.persistence import db

router = APIRouter()


def _sessao_ou_404(sessao_id: str) -> None:
    if not db.sessao_existe(sessao_id):
        raise HTTPException(status_code=404, detail=f"Sessao '{sessao_id}' nao encontrada.")


def _indice_ou_503(request: Request) -> list:
    indice = request.app.state.indice
    if not indice:
        raise HTTPException(
            status_code=503,
            detail="A base de conhecimento ainda nao foi indexada. Rode 'python -m app.ingestao' "
                   "ou use POST /api/admin/reindexar.",
        )
    return indice


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    estado = request.app.state
    return HealthResponse(
        status="ok",
        base_indexada=bool(estado.indice),
        trechos=len(estado.indice),
        indexando=estado.indexando,
        modo_geracao=config.MODO_GERACAO,
        erro_ingestao=estado.erro_ingestao,
    )


@router.post("/sessoes", response_model=NovaSessaoResponse, status_code=201)
def criar_sessao() -> NovaSessaoResponse:
    return NovaSessaoResponse(sessao_id=db.criar_sessao())


@router.post("/sessoes/{sessao_id}/mensagens", response_model=NovaMensagemResponse)
def enviar_mensagem(
    sessao_id: str,
    corpo: NovaMensagemRequest,
    request: Request,
    embedder: Embedder = Depends(dependencias.obter_embedder),
    gerador: Gerador = Depends(dependencias.obter_gerador),
) -> NovaMensagemResponse:
    _sessao_ou_404(sessao_id)
    verificar_limite(request)
    indice = _indice_ou_503(request)

    try:
        resposta = rag.responder(
            pergunta=corpo.mensagem,
            perguntas_anteriores=db.listar_perguntas_do_aluno(sessao_id),
            indice=indice,
            embedder=embedder,
            gerador=gerador,
            estado_anterior=db.obter_ultimo_estado(sessao_id),
            k=config.TOP_K,
            similaridade_minima=config.SIMILARIDADE_MINIMA,
        )
    except ChaveApiAusenteError as erro:
        raise HTTPException(status_code=503, detail=str(erro)) from erro
    except ERROS_DA_API as erro:
        raise HTTPException(status_code=502, detail=descrever_erro(erro)) from erro

    fontes = [f.model_dump() for f in resposta.fontes]
    db.registrar_mensagem(sessao_id, "aluno", corpo.mensagem)
    db.registrar_mensagem(sessao_id, "assistente", resposta.texto, fontes)
    if resposta.estado:
        db.atualizar_ultimo_estado(sessao_id, resposta.estado)

    return NovaMensagemResponse(
        resposta=resposta.texto,
        fundamentada=resposta.fundamentada,
        fontes=[FonteResponse(**f) for f in fontes],
    )


@router.get("/sessoes/{sessao_id}", response_model=HistoricoResponse)
def obter_sessao(sessao_id: str) -> HistoricoResponse:
    _sessao_ou_404(sessao_id)
    return HistoricoResponse(sessao_id=sessao_id, mensagens=db.obter_historico(sessao_id))


@router.post("/buscar", response_model=BuscaResponse)
def buscar(
    corpo: BuscaRequest,
    request: Request,
    embedder: Embedder = Depends(dependencias.obter_embedder),
) -> BuscaResponse:
    """Só a recuperação, sem geração: serve para avaliar a busca separada da resposta (Lab 4)."""
    verificar_limite(request)
    indice = _indice_ou_503(request)
    try:
        consulta, trechos = rag.recuperar(corpo.pergunta, [], indice, embedder, corpo.k)
    except ChaveApiAusenteError as erro:
        raise HTTPException(status_code=503, detail=str(erro)) from erro
    except ERROS_DA_API as erro:
        raise HTTPException(status_code=502, detail=descrever_erro(erro)) from erro
    return BuscaResponse(
        consulta=consulta,
        trechos=[
            TrechoBuscaResponse(
                fonte=t.trecho.fonte,
                secao=t.trecho.secao,
                url=t.trecho.url,
                similaridade=round(t.similaridade, 4),
                conteudo=t.trecho.conteudo,
            )
            for t in trechos
        ],
    )


@router.post("/admin/reindexar", response_model=ReindexacaoResponse, status_code=202)
def reindexar(request: Request, x_admin_token: str = Header(default="")) -> ReindexacaoResponse:
    if not config.ADMIN_TOKEN or x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token de administracao invalido.")
    if not indexacao.iniciar_reindexacao(request.app):
        raise HTTPException(status_code=409, detail="Ja existe uma indexacao em andamento.")
    return ReindexacaoResponse(status="iniciada")

"""Ponto de entrada da aplicação FastAPI.

Rode localmente com:
    uvicorn app.main:app --reload

A API fica em /api/*, a documentação interativa em /docs, e o frontend de teste
(frontend/index.html) é servido na raiz "/".
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.api import indexacao
from app.api.routers import router as api_router
from app.persistence import db

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    db.inicializar_schema()
    app.state.indice = db.carregar_indice()
    app.state.indexando = False
    app.state.erro_ingestao = ""
    # Primeira subida (ex.: deploy novo sem banco): indexa o material sozinho.
    if not app.state.indice and config.AUTO_INGESTAO and config.GEMINI_API_KEY:
        indexacao.iniciar_reindexacao(app)
    yield


app = FastAPI(
    title="Assistente da disciplina Disruptive Architectures (RAG)",
    description=(
        "Chat que responde dúvidas da disciplina usando RAG sobre o material do site: "
        "embeddings + similaridade de cosseno + Gemini, sempre citando as fontes."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# CORS aberto de propósito: projeto de estudo com um frontend de teste simples.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

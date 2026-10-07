"""Indexação em segundo plano: baixa o material, gera embeddings e recarrega o índice em memória."""

import threading

from fastapi import FastAPI

from app import ingestao
from app.api import dependencias
from app.persistence import db


def _rodar(app: FastAPI) -> None:
    try:
        ingestao.executar(dependencias.obter_embedder())
        app.state.indice = db.carregar_indice()   # troca atômica: quem está respondendo não é afetado
        app.state.erro_ingestao = ""
    except Exception as erro:  # noqa: BLE001 - qualquer falha vira status visível no /health
        app.state.erro_ingestao = str(erro)
    finally:
        app.state.indexando = False


def iniciar_reindexacao(app: FastAPI) -> bool:
    """Dispara a indexação numa thread. Retorna False se já houver uma em andamento."""
    if app.state.indexando:
        return False
    app.state.indexando = True
    threading.Thread(target=_rodar, args=(app,), daemon=True).start()
    return True

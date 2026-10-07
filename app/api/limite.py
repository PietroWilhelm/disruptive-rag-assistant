"""Limite simples de mensagens por minuto por IP (em memória).

Protege a chave do Gemini de uso abusivo quando a API está publicada.
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app import config

_acessos: dict[str, deque] = defaultdict(deque)


def ip_do_cliente(request: Request) -> str:
    encaminhado = request.headers.get("x-forwarded-for", "")
    if encaminhado:
        return encaminhado.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"


def verificar_limite(request: Request) -> None:
    agora = time.time()
    fila = _acessos[ip_do_cliente(request)]
    while fila and agora - fila[0] > 60:
        fila.popleft()
    if len(fila) >= config.LIMITE_MENSAGENS_POR_MINUTO:
        raise HTTPException(status_code=429, detail="Muitas perguntas em pouco tempo. Aguarde um minuto.")
    fila.append(agora)


def limpar() -> None:
    _acessos.clear()

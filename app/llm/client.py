"""Cliente do google-genai e wrapper de retry para a Interactions API e para embeddings.

O cliente é criado sob demanda (lazy) e cacheado — assim, subir a API, rodar os
testes ou visitar `/docs` não exige uma GEMINI_API_KEY válida.
"""

import re
import time
from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors

from app import config

try:
    # Usa o repositório de certificados do sistema operacional em vez do bundle do
    # `certifi`: evita "CERTIFICATE_VERIFY_FAILED" em redes de laboratório com
    # inspeção de TLS (o navegador funciona, o Python não).
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

# A camada de API usa isto para traduzir falhas do provedor em respostas HTTP,
# sem precisar importar o SDK do Google.
ERROS_DA_API = (genai_errors.ClientError, genai_errors.ServerError)


class CotaEsgotadaError(RuntimeError):
    """O Gemini recusou por limite de uso (cota diária ou por minuto) mesmo depois das novas tentativas."""


class ChaveApiAusenteError(RuntimeError):
    """Levantado quando algo que fala com o Gemini é usado sem GEMINI_API_KEY."""


@lru_cache(maxsize=1)
def obter_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise ChaveApiAusenteError(
            "GEMINI_API_KEY nao configurada. Defina-a no arquivo .env "
            "(veja .env.example) antes de usar o assistente."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def descrever_erro(erro) -> str:
    return f"Erro ao falar com a API do Gemini ({erro.code} {erro.status}): {erro.message}"


def _tempo_de_espera_sugerido(erro, padrao: float) -> float:
    """Usa o retryDelay sugerido pela própria API, se vier no erro."""
    try:
        detalhes = (erro.details or {}).get("error", {}).get("details", [])
        for detalhe in detalhes:
            if str(detalhe.get("@type", "")).endswith("RetryInfo"):
                achou = re.match(r"([\d.]+)s", str(detalhe.get("retryDelay", "")))
                if achou:
                    return float(achou.group(1)) + 1.0
    except Exception:
        pass
    return padrao


def cota_diaria_esgotada(erro) -> bool:
    """True se o 429 é do limite POR DIA (esperar não adianta), e não do limite por minuto."""
    texto = f"{getattr(erro, 'details', '')} {getattr(erro, 'message', '')}".lower()
    return "perday" in re.sub(r"[\s_-]", "", texto)


def _com_retry(funcao, max_tentativas: int = 5, espera_inicial: float = 5.0, **kwargs):
    """Novas tentativas com backoff exponencial para 429 (limite por minuto) e 503 (indisponível).

    Cota diária esgotada, ou 429 que persiste depois de todas as tentativas, vira
    `CotaEsgotadaError` (da camada de negócio), com uma mensagem que diz o que fazer.
    """
    espera = espera_inicial
    for tentativa in range(1, max_tentativas + 1):
        try:
            return funcao(**kwargs)
        except ERROS_DA_API as erro:
            if erro.code == 429 and cota_diaria_esgotada(erro):
                raise CotaEsgotadaError(
                    "A cota DIARIA da API do Gemini acabou (camada gratuita). Tente de novo amanha "
                    "ou ative o faturamento no Google AI Studio."
                ) from erro
            if erro.code not in (429, 503):
                raise
            if tentativa == max_tentativas:
                if erro.code == 429:
                    raise CotaEsgotadaError(
                        "A API do Gemini continuou recusando por limite de uso. Aguarde alguns minutos."
                    ) from erro
                raise
            espera = _tempo_de_espera_sugerido(erro, espera)
            print(
                f"   Erro transitorio {erro.code} ({erro.status}) - tentativa "
                f"{tentativa}/{max_tentativas}. Aguardando {espera:.0f}s..."
            )
            time.sleep(espera)
            espera *= 2


_ritmo = {"ultima_chamada": 0.0}


def respeitar_ritmo(rpm: int) -> None:
    """Espera o necessário para não passar de `rpm` chamadas por minuto (0 = sem controle)."""
    if rpm <= 0:
        return
    espera = _ritmo["ultima_chamada"] + 60.0 / rpm - time.monotonic()
    if espera > 0:
        time.sleep(espera)
    _ritmo["ultima_chamada"] = time.monotonic()


def interagir_com_retry(max_tentativas: int = 5, espera_inicial: float = 5.0, **kwargs):
    """client.interactions.create (Interactions API, Labs 2 a 4) com retry."""
    return _com_retry(obter_client().interactions.create, max_tentativas, espera_inicial, **kwargs)


def embutir_com_retry(max_tentativas: int = 5, espera_inicial: float = 5.0, **kwargs):
    """client.models.embed_content (embeddings, Lab 4) com retry e controle de ritmo."""
    embutir = obter_client().models.embed_content

    def chamar(**argumentos):
        respeitar_ritmo(config.EMBEDDING_RPM)
        return embutir(**argumentos)

    return _com_retry(chamar, max_tentativas, espera_inicial, **kwargs)

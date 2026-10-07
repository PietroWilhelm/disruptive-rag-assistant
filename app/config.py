"""Configuração central da aplicação.

Lê variáveis de ambiente (com apoio de um arquivo .env local, via python-dotenv)
e expõe valores prontos para o resto do app consumir.

A leitura da GEMINI_API_KEY NÃO falha se a chave estiver ausente: dá para subir a
API, rodar os testes e abrir o /docs sem chave. Ela só é exigida quando a camada
`llm` cria o cliente do google-genai (app/llm/client.py).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _texto(nome: str, padrao: str) -> str:
    return os.getenv(nome, padrao).strip()


def _numero(nome: str, padrao: float) -> float:
    try:
        return float(os.getenv(nome, str(padrao)))
    except ValueError:
        return padrao


def _lista(valor: str) -> list[str]:
    return [item.strip() for item in valor.split(",") if item.strip()]


def _verdadeiro(nome: str, padrao: bool) -> bool:
    return _texto(nome, str(padrao)).lower() in {"1", "true", "sim", "yes", "on"}


# --- Gemini (modelos usados nos Labs 3 e 4) ---
GEMINI_API_KEY = _texto("GEMINI_API_KEY", "")
GEMINI_MODEL = _texto("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = _texto("EMBEDDING_MODEL", "gemini-embedding-2")

# --- Banco SQLite (sessões, mensagens e índice de trechos) ---
DATABASE_PATH = _texto("DATABASE_PATH", "data/rag.db")

# --- Material da disciplina (repositório do site, fork do professor) ---
SITE_URL = _texto("SITE_URL", "https://arnaldojr.github.io/DisruptiveArchitectures/")
MATERIAL_REPO = _texto("MATERIAL_REPO", "PietroWilhelm/DisruptiveArchitectures")
MATERIAL_BRANCH = _texto("MATERIAL_BRANCH", "master")
MATERIAL_SUBDIR = _texto("MATERIAL_SUBDIR", "material")  # docs_dir do mkdocs.yml
MATERIAL_DIR = _texto("MATERIAL_DIR", "")                # pasta local já clonada (opcional)
MATERIAL_ZIP_URL = _texto("MATERIAL_ZIP_URL", "")        # sobrescreve a URL do zip (opcional)
# Filtros da indexação (separados por vírgula). INCLUIR: começos de caminho, vazio = tudo.
# EXCLUIR: padrões (* e ?) comparados com o caminho em minúsculas; o padrão ignora cópias "x copy.md".
MATERIAL_INCLUIR = _lista(_texto("MATERIAL_INCLUIR", ""))
MATERIAL_EXCLUIR = _lista(_texto("MATERIAL_EXCLUIR", "*copy.*"))

# Ritmo máximo de chamadas de embedding por minuto (camada gratuita: 100). 0 desliga o controle.
EMBEDDING_RPM = int(_numero("EMBEDDING_RPM", 80))

# --- Recuperação ---
TOP_K = int(_numero("TOP_K", 4))
SIMILARIDADE_MINIMA = _numero("SIMILARIDADE_MINIMA", 0.40)

# --- Operação ---
AUTO_INGESTAO = _verdadeiro("AUTO_INGESTAO", True)
LIMITE_MENSAGENS_POR_MINUTO = int(_numero("LIMITE_MENSAGENS_POR_MINUTO", 20))


def caminho_absoluto_banco() -> str:
    """Resolve DATABASE_PATH relativo à raiz do projeto, criando a pasta se necessário."""
    caminho = Path(DATABASE_PATH)
    if not caminho.is_absolute():
        caminho = BASE_DIR / caminho
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return str(caminho)


def url_do_zip_do_material() -> str:
    if MATERIAL_ZIP_URL:
        return MATERIAL_ZIP_URL
    return f"https://github.com/{MATERIAL_REPO}/archive/refs/heads/{MATERIAL_BRANCH}.zip"

"""Garante, lendo o código, que a camada de negócio não depende de IA, banco, HTTP ou das outras camadas."""

import ast
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "app" / "business"
PROIBIDOS = ("google", "genai", "fastapi", "starlette", "sqlite3", "httpx", "dotenv", "truststore", "app.llm",
             "app.api", "app.persistence", "app.config", "app.ingestao")


def _imports(arquivo: Path) -> list[str]:
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    nomes = []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes += [a.name for a in no.names]
        elif isinstance(no, ast.ImportFrom):
            nomes.append(no.module or "")
    return nomes


def test_business_nao_importa_nada_de_ia_banco_ou_http():
    arquivos = sorted(PASTA.glob("*.py"))
    assert arquivos
    for arquivo in arquivos:
        for nome in _imports(arquivo):
            assert not nome.startswith(PROIBIDOS), f"{arquivo.name} importa {nome}"


def test_nenhum_arquivo_do_projeto_usa_future_annotations():
    raiz = PASTA.parent.parent
    ignorados = {".venv", "__pycache__"}
    for arquivo in raiz.rglob("*.py"):
        if ignorados & set(arquivo.parts) or arquivo.name == "test_independencia.py":
            continue
        assert "from __future__" not in arquivo.read_text(encoding="utf-8"), arquivo

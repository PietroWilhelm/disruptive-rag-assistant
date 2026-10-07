"""Ingestão do material da disciplina: arquivos .md/.ipynb -> trechos -> embeddings -> SQLite.

A fonte é o repositório do site (o fork no GitHub): o `docs_dir` do mkdocs.yml é a pasta
`material/`. Dois jeitos de informar de onde ler:

  1. MATERIAL_DIR=/caminho/do/clone   -> usa a pasta local (rápido para desenvolver);
  2. sem MATERIAL_DIR                 -> baixa o zip do repositório (MATERIAL_REPO/BRANCH).

Rode com:  python -m app.ingestao

É retomável: cada trecho tem um hash (conteúdo + modelo de embedding). Trecho que já está
no banco não é embutido de novo, e trecho que sumiu do material é removido do índice.
"""

import hashlib
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.business import conhecimento
from app.business.portas import Embedder
from app.persistence import db

PASTAS_IGNORADAS = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints"}


@contextmanager
def obter_pasta_do_material():
    """Entrega a pasta com os arquivos do material (local ou baixada do GitHub)."""
    if config.MATERIAL_DIR:
        pasta = Path(config.MATERIAL_DIR).expanduser()
        if not pasta.is_dir():
            raise FileNotFoundError(f"MATERIAL_DIR nao encontrada: {pasta}")
        yield pasta
        return

    url = config.url_do_zip_do_material()
    with tempfile.TemporaryDirectory() as temporario:
        destino = Path(temporario)
        arquivo_zip = destino / "material.zip"
        print(f"Baixando {url} ...")
        with urllib.request.urlopen(url, timeout=120) as resposta:
            arquivo_zip.write_bytes(resposta.read())
        with zipfile.ZipFile(arquivo_zip) as zip_:
            raiz_segura = destino.resolve()
            for nome in zip_.namelist():
                if not (destino / nome).resolve().is_relative_to(raiz_segura):
                    raise ValueError(f"Caminho suspeito no zip: {nome}")
            zip_.extractall(destino / "repo")
        raizes = [p for p in (destino / "repo").iterdir() if p.is_dir()]
        base = raizes[0] if len(raizes) == 1 else destino / "repo"
        pasta = base / config.MATERIAL_SUBDIR
        if not pasta.is_dir():
            raise FileNotFoundError(f"Pasta '{config.MATERIAL_SUBDIR}' nao existe no repositorio baixado.")
        yield pasta

def no_escopo(fonte: str) -> bool:
    """True se o arquivo está dentro de MATERIAL_INCLUIR (vazio = tudo). Só olha o INCLUIR, não o EXCLUIR."""
    return not config.MATERIAL_INCLUIR or any(fonte.startswith(prefixo) for prefixo in config.MATERIAL_INCLUIR)

def listar_arquivos(pasta: Path) -> list[Path]:
    return sorted(
        arquivo
        for arquivo in pasta.rglob("*")
        if arquivo.is_file()
        and arquivo.suffix.lower() in conhecimento.EXTENSOES_SUPORTADAS
        and not (PASTAS_IGNORADAS & set(arquivo.relative_to(pasta).parts))
    )


def hash_do_trecho(trecho) -> str:
    base = "\x1f".join([config.EMBEDDING_MODEL, trecho.fonte, trecho.secao, trecho.conteudo])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def executar(embedder: Embedder, progresso=print) -> dict:
    """Atualiza o índice no banco. Retorna um resumo (arquivos, trechos, novos, removidos)."""
    db.inicializar_schema()
    removidos = db.remover_trechos_exceto(hashes_atuais, no_escopo)
    ja_indexados = db.hashes_dos_trechos()
    hashes_atuais: set[str] = set()
    arquivos_lidos = novos = 0

    with obter_pasta_do_material() as pasta:
        for arquivo in listar_arquivos(pasta):
            fonte = arquivo.relative_to(pasta).as_posix()
            try:
                trechos = conhecimento.segmentar_arquivo(
                    fonte, arquivo.read_text(encoding="utf-8"), config.SITE_URL
                )
            except (ValueError, UnicodeDecodeError) as erro:
                progresso(f"  ! {fonte}: ignorado ({erro})")
                continue
            arquivos_lidos += 1
            for trecho in trechos:
                h = hash_do_trecho(trecho)
                hashes_atuais.add(h)
                if h in ja_indexados:
                    continue
                vetor = embedder.embutir_documento(trecho.secao, trecho.conteudo)
                db.salvar_trecho(h, trecho, vetor)
                novos += 1
            progresso(f"  {fonte}: {len(trechos)} trechos")

    if not hashes_atuais:
        raise RuntimeError("Nenhum trecho gerado: o material esta vazio ou fora do formato esperado.")

    removidos = db.remover_trechos_exceto(hashes_atuais)
    db.definir_metadado("indexado_em", datetime.now(timezone.utc).isoformat())
    resumo = {
        "arquivos": arquivos_lidos,
        "trechos": len(hashes_atuais),
        "novos": novos,
        "removidos": removidos,
    }
    progresso(f"Indice atualizado: {resumo}")
    return resumo


if __name__ == "__main__":
    from app.llm.embeddings import GeminiEmbedder

    executar(GeminiEmbedder())

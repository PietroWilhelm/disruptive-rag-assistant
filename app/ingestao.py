"""Ingestão do material da disciplina: arquivos .md/.ipynb -> trechos -> embeddings -> SQLite.

A fonte é o repositório do site (o fork no GitHub): o `docs_dir` do mkdocs.yml é a pasta
`material/`. Dois jeitos de informar de onde ler:

  1. MATERIAL_DIR=/caminho/do/clone   -> usa a pasta local (rápido para desenvolver);
  2. sem MATERIAL_DIR                 -> baixa o zip do repositório (MATERIAL_REPO/BRANCH).

Rode com:  python -m app.ingestao            (indexa)
           python -m app.ingestao --contar   (só conta quantos embeddings faltam, sem chamar a API)

É retomável: cada trecho tem um hash (conteúdo + modelo de embedding). Trecho que já está
no banco não é embutido de novo, e trecho que sumiu do material é removido do índice
(só entre os arquivos dentro de MATERIAL_INCLUIR: o que está fora do filtro não é tocado).
Se a cota da API acabar no meio, o que já foi indexado fica salvo e nada é removido: é só
rodar de novo mais tarde (ou no dia seguinte, se for a cota diária).

Filtros (variáveis de ambiente, separadas por vírgula):
  MATERIAL_INCLUIR=aulas/genAI,aulas/checkpoint,agenda   só estes começos de caminho
  MATERIAL_EXCLUIR=*copy.*,*cp-correcao*                 estes padrões ficam de fora
"""

import fnmatch
import hashlib
import tempfile
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.business import conhecimento
from app.llm import embeddings
from app.llm.client import CotaEsgotadaError
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


def deve_indexar(fonte: str) -> bool:
    """Aplica MATERIAL_INCLUIR (começos de caminho) e MATERIAL_EXCLUIR (padrões com * e ?)."""
    if config.MATERIAL_INCLUIR and not any(fonte.startswith(prefixo) for prefixo in config.MATERIAL_INCLUIR):
        return False
    return not any(fnmatch.fnmatch(fonte.lower(), padrao.lower()) for padrao in config.MATERIAL_EXCLUIR)


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


def executar(progresso=print) -> dict:
    """Atualiza o índice no banco. Retorna um resumo (arquivos, trechos, novos, removidos)."""
    db.inicializar_schema()
    ja_indexados = db.hashes_dos_trechos()
    hashes_atuais: set[str] = set()
    arquivos_lidos = novos = ignorados = 0

    with obter_pasta_do_material() as pasta:
        for arquivo in listar_arquivos(pasta):
            fonte = arquivo.relative_to(pasta).as_posix()
            if not deve_indexar(fonte):
                ignorados += 1
                continue
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
                try:
                    vetor = embeddings.embutir_documento(trecho.secao, trecho.conteudo)
                except CotaEsgotadaError:
                    progresso(
                        f"\nCota da API esgotada em '{fonte}'. {novos} trechos novos ja estao salvos; "
                        "nada foi removido. Rode o mesmo comando de novo mais tarde (ou amanha, se for a "
                        "cota diaria): ele continua de onde parou."
                    )
                    raise
                db.salvar_trecho(h, trecho, vetor)
                novos += 1
            progresso(f"  {fonte}: {len(trechos)} trechos")

    if not hashes_atuais:
        raise RuntimeError("Nenhum trecho gerado: o material esta vazio ou fora do formato esperado.")

    removidos = db.remover_trechos_exceto(hashes_atuais, no_escopo)
    db.definir_metadado("indexado_em", datetime.now(timezone.utc).isoformat())
    resumo = {
        "arquivos": arquivos_lidos,
        "trechos": len(hashes_atuais),
        "novos": novos,
        "removidos": removidos,
        "arquivos_ignorados_pelo_filtro": ignorados,
    }
    progresso(f"Indice atualizado: {resumo}")
    return resumo


def contar_pendentes(progresso=print) -> dict:
    """Quantos trechos o material tem (com os filtros) e quantos ainda precisam de embedding.
    Não chama a API: serve para planejar a indexação dentro da cota."""
    db.inicializar_schema()
    ja_indexados = db.hashes_dos_trechos()
    por_pasta: dict[str, list[int]] = {}

    with obter_pasta_do_material() as pasta:
        for arquivo in listar_arquivos(pasta):
            fonte = arquivo.relative_to(pasta).as_posix()
            if not deve_indexar(fonte):
                continue
            try:
                trechos = conhecimento.segmentar_arquivo(
                    fonte, arquivo.read_text(encoding="utf-8"), config.SITE_URL
                )
            except (ValueError, UnicodeDecodeError):
                continue
            partes = fonte.split("/")
            grupo = "/".join(partes[:3]) if len(partes) > 3 else "/".join(partes[:-1]) or partes[0]
            contagem = por_pasta.setdefault(grupo, [0, 0])
            contagem[0] += len(trechos)
            contagem[1] += sum(1 for t in trechos if hash_do_trecho(t) not in ja_indexados)

    total = sum(c[0] for c in por_pasta.values())
    faltam = sum(c[1] for c in por_pasta.values())
    for grupo, (n_total, n_faltam) in sorted(por_pasta.items()):
        progresso(f"  {grupo:<28} {n_total:>5} trechos  |  faltam {n_faltam:>5}")
    progresso(f"Total: {total} trechos; faltam {faltam} embeddings (1 requisicao cada).")
    return {"trechos": total, "faltam": faltam, "por_pasta": por_pasta}


if __name__ == "__main__":
    import sys

    if "--contar" in sys.argv:
        contar_pendentes()
        sys.exit(0)

    try:
        executar()
    except CotaEsgotadaError as erro:
        print(f"Parou: {erro}")
        sys.exit(1)

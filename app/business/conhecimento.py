"""Base de conhecimento: transforma os arquivos do material (.md e .ipynb) em trechos.

Funções puras: recebem texto e devolvem `Trecho`. Ler arquivos do disco, baixar o
repositório e gerar embeddings são tarefas de `app/ingestao.py`.

Cada trecho guarda a `fonte` no formato do Lab 4 (nome do arquivo, ex.: lab4.md),
mas com o caminho relativo ao material para não confundir arquivos homônimos
(aulas/iot/lab1.md x aulas/genAI/lab1/lab1.md).
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import quote

from app.business.schemas import Trecho

EXTENSOES_SUPORTADAS = (".md", ".ipynb")
LIMITE_PADRAO = 1200   # caracteres por trecho
TAMANHO_MINIMO = 20    # descarta trechos praticamente vazios

_RE_TITULO = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_RE_CERCA = re.compile(r"^\s*(```|~~~)")
_RE_COMENTARIO_HTML = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_IMAGEM = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


@dataclass
class _Secao:
    caminho: list[str]
    ancora: str
    linhas: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ URLs e âncoras
def slug_ancora(titulo: str) -> str:
    """Mesma regra de âncora que o MkDocs (Python-Markdown) usa nos títulos."""
    texto = unicodedata.normalize("NFKD", titulo).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^\w\s-]", "", texto).strip().lower()
    return re.sub(r"[-\s]+", "-", texto)


def url_da_fonte(fonte: str, url_base: str, ancora: str = "") -> str:
    """aulas/genAI/lab4/lab4.md -> <url_base>/aulas/genAI/lab4/lab4/#ancora"""
    partes = list(PurePosixPath(fonte).with_suffix("").parts)
    if partes and partes[-1] == "index":
        partes = partes[:-1]
    url = url_base.rstrip("/") + "/"
    if partes:
        url += "/".join(quote(p) for p in partes) + "/"
    return f"{url}#{ancora}" if ancora else url


# ------------------------------------------------------------------ leitura dos formatos
def remover_front_matter(texto: str) -> str:
    if texto.startswith("---\n"):
        fim = texto.find("\n---", 4)
        if fim != -1:
            return texto[fim + 4:].lstrip("\n")
    return texto


def texto_de_notebook(conteudo_json: str) -> str:
    """Notebook -> markdown: células de texto viram texto, células de código viram blocos de código.
    As saídas (outputs) são ignoradas."""
    notebook = json.loads(conteudo_json)
    partes = []
    for celula in notebook.get("cells", []):
        origem = celula.get("source", "")
        if isinstance(origem, list):
            origem = "".join(origem)
        if not origem.strip():
            continue
        if celula.get("cell_type") == "markdown":
            partes.append(origem.strip())
        elif celula.get("cell_type") == "code":
            partes.append("```python\n" + origem.strip("\n") + "\n```")
    return "\n\n".join(partes)


def _limpar_titulo(titulo: str) -> str:
    titulo = re.sub(r"\{#[^}]*\}", "", titulo)                 # {#id}
    titulo = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", titulo)   # [texto](link)
    return re.sub(r"[`*]", "", titulo).strip()


# ------------------------------------------------------------------ seções e blocos
def dividir_em_secoes(texto: str) -> list[_Secao]:
    """Divide pelos títulos (#, ##, ...), ignorando '#' dentro de blocos de código."""
    secoes = [_Secao(caminho=[], ancora="")]
    pilha: list[tuple[int, str]] = []
    dentro_de_cerca = False

    for linha in texto.splitlines():
        if _RE_CERCA.match(linha):
            dentro_de_cerca = not dentro_de_cerca
        achou = None if dentro_de_cerca else _RE_TITULO.match(linha)
        if achou:
            nivel, titulo = len(achou.group(1)), _limpar_titulo(achou.group(2))
            while pilha and pilha[-1][0] >= nivel:
                pilha.pop()
            pilha.append((nivel, titulo))
            secoes.append(_Secao(caminho=[t for _, t in pilha], ancora=slug_ancora(titulo)))
        else:
            secoes[-1].linhas.append(linha)
    return secoes


def _blocos(linhas: list[str]) -> list[str]:
    """Parágrafos (separados por linha em branco); um bloco de código é um bloco único."""
    blocos: list[str] = []
    atual: list[str] = []
    dentro_de_cerca = False
    for linha in linhas:
        if _RE_CERCA.match(linha):
            if not dentro_de_cerca and atual:       # a cerca abre um novo bloco
                blocos.append("\n".join(atual))
                atual = []
            atual.append(linha)
            dentro_de_cerca = not dentro_de_cerca
            if not dentro_de_cerca:                 # a cerca fechou: fecha o bloco
                blocos.append("\n".join(atual))
                atual = []
        elif dentro_de_cerca:
            atual.append(linha)
        elif linha.strip() == "":
            if atual:
                blocos.append("\n".join(atual))
                atual = []
        else:
            atual.append(linha)
    if atual:
        blocos.append("\n".join(atual))
    return [b.strip("\n") for b in blocos if b.strip()]


def _limpar_bloco(bloco: str) -> str:
    if _RE_CERCA.match(bloco):
        return bloco
    bloco = _RE_COMENTARIO_HTML.sub("", bloco)
    bloco = _RE_IMAGEM.sub(lambda m: f"[imagem: {m.group(1)}]" if m.group(1) else "", bloco)
    return bloco.strip()


def _dividir_bloco(bloco: str, limite: int) -> list[str]:
    """Quebra um bloco maior que o limite, mantendo cada pedaço de código com suas cercas."""
    if len(bloco) <= limite:
        return [bloco]
    linhas = bloco.split("\n")
    eh_codigo = bool(_RE_CERCA.match(bloco))
    abertura = linhas[0] if eh_codigo else ""
    if eh_codigo:
        linhas = linhas[1:-1] if len(linhas) > 1 and _RE_CERCA.match(linhas[-1]) else linhas[1:]
    reserva = len(abertura) + 5 if eh_codigo else 0

    pedacos, atual, tamanho = [], [], 0
    for linha in linhas:
        # linha isolada maior que o limite: corta no limite
        while len(linha) > limite - reserva:
            if atual:
                pedacos.append(atual)
                atual, tamanho = [], 0
            pedacos.append([linha[: limite - reserva]])
            linha = linha[limite - reserva:]
        if atual and tamanho + len(linha) + 1 > limite - reserva:
            pedacos.append(atual)
            atual, tamanho = [], 0
        atual.append(linha)
        tamanho += len(linha) + 1
    if atual:
        pedacos.append(atual)

    if eh_codigo:
        return [f"{abertura}\n" + "\n".join(p) + "\n```" for p in pedacos]
    return ["\n".join(p) for p in pedacos]


def _empacotar(blocos: list[str], limite: int) -> list[str]:
    grupos, atual = [], ""
    for bloco in blocos:
        for pedaco in _dividir_bloco(bloco, limite):
            if atual and len(atual) + len(pedaco) + 2 > limite:
                grupos.append(atual)
                atual = ""
            atual = f"{atual}\n\n{pedaco}" if atual else pedaco
    if atual:
        grupos.append(atual)
    return grupos


# ------------------------------------------------------------------ API pública
def titulo_do_documento(texto: str, fonte: str) -> str:
    for linha in texto.splitlines():
        achou = _RE_TITULO.match(linha)
        if achou and len(achou.group(1)) == 1:
            return _limpar_titulo(achou.group(2))
    nome = PurePosixPath(fonte).stem
    return nome.replace("_", " ").replace("-", " ").strip().title()


def segmentar(fonte: str, texto: str, url_base: str, limite: int = LIMITE_PADRAO) -> list[Trecho]:
    """Markdown -> trechos, uma ou mais por seção, cada um com fonte, título, seção e link."""
    texto = remover_front_matter(texto)
    titulo_doc = titulo_do_documento(texto, fonte)
    trechos: list[Trecho] = []

    for secao in dividir_em_secoes(texto):
        caminho = secao.caminho
        if not caminho or caminho[0] != titulo_doc:
            caminho = [titulo_doc] + caminho
        nome_secao = " > ".join(caminho)

        blocos = [b for b in (_limpar_bloco(b) for b in _blocos(secao.linhas)) if b]
        for grupo in _empacotar(blocos, limite):
            if len(grupo.strip()) < TAMANHO_MINIMO:
                continue
            trechos.append(
                Trecho(
                    fonte=fonte,
                    titulo=titulo_doc,
                    secao=nome_secao,
                    url=url_da_fonte(fonte, url_base, secao.ancora),
                    conteudo=grupo.strip(),
                )
            )
    return trechos


def segmentar_arquivo(fonte: str, conteudo_bruto: str, url_base: str, limite: int = LIMITE_PADRAO) -> list[Trecho]:
    """Escolhe o leitor pela extensão (.md ou .ipynb) e segmenta."""
    extensao = PurePosixPath(fonte).suffix.lower()
    if extensao == ".ipynb":
        texto = texto_de_notebook(conteudo_bruto)
    elif extensao == ".md":
        texto = conteudo_bruto
    else:
        raise ValueError(f"Extensão não suportada: {extensao}")
    return segmentar(fonte, texto, url_base, limite)

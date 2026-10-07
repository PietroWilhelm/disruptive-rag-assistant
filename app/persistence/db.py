"""Persistência em SQLite (sem ORM, uma conexão por operação — mesmo estilo do petshop-assistant).

Guarda o que o Lab 3.5 pede como "informações relevantes da aplicação":
- sessões de conversa e o último `interaction_id` (memória da conversa),
- mensagens trocadas, com as fontes usadas em cada resposta (logs),
- o índice de conhecimento: trechos do material + embeddings (BLOB float32).
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

import numpy as np

from app import config
from app.business.schemas import ItemIndice, Trecho

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessoes (
    id TEXT PRIMARY KEY,
    criado_em TEXT NOT NULL,
    ultimo_estado TEXT
);

CREATE TABLE IF NOT EXISTS mensagens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sessao_id TEXT NOT NULL REFERENCES sessoes(id),
    remetente TEXT NOT NULL CHECK (remetente IN ('aluno', 'assistente')),
    texto TEXT NOT NULL,
    fontes_json TEXT NOT NULL DEFAULT '[]',
    criado_em TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trechos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT NOT NULL UNIQUE,
    fonte TEXT NOT NULL,
    titulo TEXT NOT NULL,
    secao TEXT NOT NULL,
    url TEXT NOT NULL,
    conteudo TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS metadados (
    chave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);
"""


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conectar() -> sqlite3.Connection:
    conexao = sqlite3.connect(config.caminho_absoluto_banco())
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA foreign_keys = ON;")
    return conexao


def inicializar_schema() -> None:
    with closing(_conectar()) as conexao, conexao:
        conexao.executescript(_SCHEMA)


# ------------------------------------------------------------------ sessões e mensagens
def criar_sessao() -> str:
    sessao_id = str(uuid.uuid4())
    with closing(_conectar()) as conexao, conexao:
        conexao.execute(
            "INSERT INTO sessoes (id, criado_em, ultimo_estado) VALUES (?, ?, NULL)",
            (sessao_id, _agora()),
        )
    return sessao_id


def sessao_existe(sessao_id: str) -> bool:
    with closing(_conectar()) as conexao:
        linha = conexao.execute("SELECT 1 FROM sessoes WHERE id = ?", (sessao_id,)).fetchone()
    return linha is not None


def obter_ultimo_estado(sessao_id: str) -> str | None:
    with closing(_conectar()) as conexao:
        linha = conexao.execute(
            "SELECT ultimo_estado FROM sessoes WHERE id = ?", (sessao_id,)
        ).fetchone()
    return linha["ultimo_estado"] if linha else None


def atualizar_ultimo_estado(sessao_id: str, estado: str) -> None:
    with closing(_conectar()) as conexao, conexao:
        conexao.execute("UPDATE sessoes SET ultimo_estado = ? WHERE id = ?", (estado, sessao_id))


def registrar_mensagem(sessao_id: str, remetente: str, texto: str, fontes: list[dict] | None = None) -> None:
    with closing(_conectar()) as conexao, conexao:
        conexao.execute(
            "INSERT INTO mensagens (sessao_id, remetente, texto, fontes_json, criado_em) VALUES (?, ?, ?, ?, ?)",
            (sessao_id, remetente, texto, json.dumps(fontes or [], ensure_ascii=False), _agora()),
        )


def listar_perguntas_do_aluno(sessao_id: str, limite: int = 3) -> list[str]:
    """Últimas perguntas do aluno, da mais antiga para a mais recente."""
    with closing(_conectar()) as conexao:
        linhas = conexao.execute(
            "SELECT texto FROM mensagens WHERE sessao_id = ? AND remetente = 'aluno' ORDER BY id DESC LIMIT ?",
            (sessao_id, limite),
        ).fetchall()
    return [linha["texto"] for linha in reversed(linhas)]


def obter_historico(sessao_id: str) -> list[dict]:
    with closing(_conectar()) as conexao:
        linhas = conexao.execute(
            "SELECT remetente, texto, fontes_json, criado_em FROM mensagens WHERE sessao_id = ? ORDER BY id",
            (sessao_id,),
        ).fetchall()
    mensagens = []
    for linha in linhas:
        dado = dict(linha)
        dado["fontes"] = json.loads(dado.pop("fontes_json"))
        mensagens.append(dado)
    return mensagens


# ------------------------------------------------------------------ índice de conhecimento
def hashes_dos_trechos() -> set[str]:
    with closing(_conectar()) as conexao:
        return {linha["hash"] for linha in conexao.execute("SELECT hash FROM trechos")}


def salvar_trecho(hash_trecho: str, trecho: Trecho, embedding: np.ndarray) -> None:
    vetor = np.asarray(embedding, dtype=np.float32).tobytes()
    with closing(_conectar()) as conexao, conexao:
        conexao.execute(
            """INSERT OR REPLACE INTO trechos (hash, fonte, titulo, secao, url, conteudo, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (hash_trecho, trecho.fonte, trecho.titulo, trecho.secao, trecho.url, trecho.conteudo, vetor),
        )


def remover_trechos_exceto(hashes_para_manter: set[str], no_escopo=None) -> int:
    """Apaga do índice os trechos que não existem mais no material. Retorna quantos foram removidos.

    `no_escopo(fonte) -> bool` limita a limpeza aos arquivos que a ingestão olhou nesta rodada
    (None = todo o índice). Trechos de arquivos fora do escopo ficam como estão.
    """
    with closing(_conectar()) as conexao:
        linhas = conexao.execute("SELECT hash, fonte FROM trechos").fetchall()
    obsoletos = [
        linha["hash"]
        for linha in linhas
        if linha["hash"] not in hashes_para_manter and (no_escopo is None or no_escopo(linha["fonte"]))
    ]
    with closing(_conectar()) as conexao, conexao:
        conexao.executemany("DELETE FROM trechos WHERE hash = ?", [(h,) for h in obsoletos])
    return len(obsoletos)


def carregar_indice() -> list[ItemIndice]:
    with closing(_conectar()) as conexao:
        linhas = conexao.execute(
            "SELECT fonte, titulo, secao, url, conteudo, embedding FROM trechos ORDER BY id"
        ).fetchall()
    return [
        ItemIndice(
            trecho=Trecho(
                fonte=linha["fonte"],
                titulo=linha["titulo"],
                secao=linha["secao"],
                url=linha["url"],
                conteudo=linha["conteudo"],
            ),
            embedding=np.frombuffer(linha["embedding"], dtype=np.float32),
        )
        for linha in linhas
    ]


def contar_trechos() -> int:
    with closing(_conectar()) as conexao:
        return conexao.execute("SELECT COUNT(*) AS n FROM trechos").fetchone()["n"]


def definir_metadado(chave: str, valor: str) -> None:
    with closing(_conectar()) as conexao, conexao:
        conexao.execute("INSERT OR REPLACE INTO metadados (chave, valor) VALUES (?, ?)", (chave, valor))


def obter_metadado(chave: str) -> str | None:
    with closing(_conectar()) as conexao:
        linha = conexao.execute("SELECT valor FROM metadados WHERE chave = ?", (chave,)).fetchone()
    return linha["valor"] if linha else None

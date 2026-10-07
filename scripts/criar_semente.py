"""Cria seed/indice.db a partir do banco local: só os trechos e os embeddings, sem sessões nem mensagens.

    python -m scripts.criar_semente

Esse arquivo vai para o git junto com o código. No deploy, se o banco estiver vazio, o app copia o
índice dele para o banco e não precisa gastar a cota do Gemini reindexando o material.
Rode de novo (e commite) sempre que o site mudar e você reindexar.
"""

from pathlib import Path

from app import config
from app.persistence import db

DESTINO = Path(__file__).resolve().parent.parent / "seed" / "indice.db"


def main() -> None:
    db.inicializar_schema()
    if not db.contar_trechos():
        raise SystemExit("O banco local esta vazio. Rode antes: python -m app.ingestao")
    total = db.exportar_indice(str(DESTINO))
    megabytes = DESTINO.stat().st_size / 1_000_000
    print(f"{total} trechos gravados em {DESTINO} ({megabytes:.1f} MB), modelo de embedding: {config.EMBEDDING_MODEL}.")


if __name__ == "__main__":
    main()

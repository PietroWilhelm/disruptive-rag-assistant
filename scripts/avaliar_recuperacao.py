"""Avalia a RECUPERAÇÃO separada da geração (Lab 4) usando o índice já gravado no banco.

    python -m scripts.avaliar_recuperacao              # só busca (barato: 1 embedding por pergunta)
    python -m scripts.avaliar_recuperacao --respostas  # também gera as respostas e confere as citações

O que mede:
  - acerto@k  : em quantas perguntas pelo menos um trecho esperado aparece entre os k recuperados;
  - MRR       : quão no topo o primeiro trecho esperado aparece;
  - similaridade: das perguntas do escopo x fora do escopo, para escolher SIMILARIDADE_MINIMA.

Edite avaliacao/perguntas.json com perguntas reais da disciplina ("fontes_esperadas" é um
pedaço do caminho do arquivo, ex.: "genAI/lab4").
"""

import argparse
import json
import statistics
from pathlib import Path

from app import config
from app.business import rag
from app.llm.assistente import GeminiGerador
from app.llm.embeddings import GeminiEmbedder
from app.persistence import db

ARQUIVO = Path(__file__).resolve().parent.parent / "avaliacao" / "perguntas.json"


def main(com_respostas: bool) -> None:
    db.inicializar_schema()
    indice = db.carregar_indice()
    if not indice:
        raise SystemExit("Indice vazio. Rode antes: python -m app.ingestao")

    perguntas = json.loads(ARQUIVO.read_text(encoding="utf-8"))
    embedder, gerador, k = GeminiEmbedder(), GeminiGerador(), config.TOP_K

    acertos, reciprocos, melhores = 0, [], []
    print(f"\n=== Recuperacao (k={k}) ===")
    for item in perguntas["dentro_do_escopo"]:
        _, trechos = rag.recuperar(item["pergunta"], [], indice, embedder, k)
        posicao = next(
            (i for i, t in enumerate(trechos, start=1)
             if any(esperada in t.trecho.fonte for esperada in item["fontes_esperadas"])),
            None,
        )
        acertos += posicao is not None
        reciprocos.append(1 / posicao if posicao else 0.0)
        melhores.append(trechos[0].similaridade)
        marca = f"OK (posicao {posicao})" if posicao else "ERRO"
        print(f"  {marca:<16} sim={trechos[0].similaridade:.3f}  {item['pergunta']}")
        if not posicao:
            print(f"      recuperou: {[t.trecho.fonte for t in trechos]}")

    total = len(perguntas["dentro_do_escopo"])
    print(f"\nacerto@{k}: {acertos}/{total} ({acertos / total:.0%})   MRR: {sum(reciprocos) / total:.3f}")

    print("\n=== Fora do escopo (a similaridade deveria ser baixa) ===")
    fora = []
    for pergunta in perguntas["fora_do_escopo"]:
        _, trechos = rag.recuperar(pergunta, [], indice, embedder, 1)
        fora.append(trechos[0].similaridade)
        print(f"  sim={trechos[0].similaridade:.3f}  {pergunta}")

    print(
        f"\nSimilaridade do melhor trecho: no escopo min={min(melhores):.3f} media={statistics.mean(melhores):.3f} | "
        f"fora do escopo max={max(fora):.3f}\n"
        f"Escolha SIMILARIDADE_MINIMA entre os dois (atual: {config.SIMILARIDADE_MINIMA})."
    )

    if com_respostas:
        print("\n=== Respostas (confere citacoes e recusa) ===")
        for item in perguntas["dentro_do_escopo"]:
            r = rag.responder(item["pergunta"], [], indice, embedder, gerador, None, k, config.SIMILARIDADE_MINIMA)
            citou = any(f.citada for f in r.fontes)
            print(f"  fundamentada={r.fundamentada!s:<5} citou_fonte={citou!s:<5} "
                  f"citacoes_invalidas={r.citacoes_invalidas}  {item['pergunta']}")
        for pergunta in perguntas["fora_do_escopo"]:
            r = rag.responder(pergunta, [], indice, embedder, gerador, None, k, config.SIMILARIDADE_MINIMA)
            print(f"  recusou={not r.fundamentada or r.texto.startswith('Não encontrei')!s:<5}  {pergunta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--respostas", action="store_true")
    main(parser.parse_args().respostas)

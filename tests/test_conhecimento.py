import json

import pytest

from app.business import conhecimento as c
from tests.conftest import CHECKPOINT_IPYNB, LAB4_MD

BASE = "https://exemplo.github.io/Site/"


def test_slug_ancora_igual_ao_mkdocs():
    assert c.slug_ancora("Memória da Conversa") == "memoria-da-conversa"
    assert c.slug_ancora("Lab 4 - RAG e bases") == "lab-4-rag-e-bases"
    assert c.slug_ancora("1. Instalação (SDK)") == "1-instalacao-sdk"


def test_url_da_fonte():
    assert c.url_da_fonte("aulas/genAI/lab4/lab4.md", BASE) == BASE + "aulas/genAI/lab4/lab4/"
    assert c.url_da_fonte("aulas/IA/lab01/dataframe copy.ipynb", BASE, "x") == BASE + "aulas/IA/lab01/dataframe%20copy/#x"
    assert c.url_da_fonte("index.md", BASE) == BASE
    assert c.url_da_fonte("aulas/iot/index.md", BASE) == BASE + "aulas/iot/"


def test_front_matter_removido():
    assert c.remover_front_matter("---\ntitle: x\n---\n# A\ntexto").startswith("# A")
    assert c.remover_front_matter("# A") == "# A"


def test_titulo_dentro_de_bloco_de_codigo_nao_vira_secao():
    secoes = c.dividir_em_secoes("# Doc\n\n## Código\n\n```python\n# comentário\nx = 1\n```\n")
    nomes = [s.caminho for s in secoes]
    assert ["Doc", "Código"] in nomes
    assert not any("comentário" in " ".join(n) for n in nomes)


def test_segmentar_markdown_guarda_fonte_secao_e_link():
    trechos = c.segmentar("aulas/genAI/lab4/lab4.md", LAB4_MD, BASE)
    por_secao = {t.secao: t for t in trechos}
    cosseno = por_secao["Lab 4 - RAG e bases de conhecimento > Similaridade de cosseno"]
    assert cosseno.fonte == "aulas/genAI/lab4/lab4.md"
    assert cosseno.titulo == "Lab 4 - RAG e bases de conhecimento"
    assert cosseno.url == BASE + "aulas/genAI/lab4/lab4/#similaridade-de-cosseno"
    assert "# este # nao e um titulo" in cosseno.conteudo
    assert "title: Lab 4" not in " ".join(t.conteudo for t in trechos)


def test_notebook_vira_texto_sem_saidas():
    texto = c.texto_de_notebook(json.dumps(CHECKPOINT_IPYNB))
    assert "# Checkpoint 4 - Pet Shop" in texto
    assert "```python\nfrom pydantic import BaseModel" in texto
    assert "SAIDA IGNORADA" not in texto


def test_segmentar_arquivo_notebook_e_extensao_invalida():
    trechos = c.segmentar_arquivo("aulas/checkpoint/cp4.ipynb", json.dumps(CHECKPOINT_IPYNB), BASE)
    assert any("Guardrail veterinário" in t.secao for t in trechos)
    with pytest.raises(ValueError):
        c.segmentar_arquivo("foto.png", "x", BASE)


def test_limite_respeitado_e_codigo_continua_valido():
    codigo = "```python\n" + "\n".join(f"linha_{i} = {i}" for i in range(300)) + "\n```"
    texto = "# Doc\n\n## Código longo\n\nIntrodução do código.\n\n" + codigo
    trechos = c.segmentar("a/b.md", texto, BASE, limite=400)
    assert len(trechos) > 3
    assert all(len(t.conteudo) <= 400 for t in trechos)
    for t in trechos:
        if "linha_" in t.conteudo:
            assert t.conteudo.count("```") % 2 == 0   # cercas sempre pareadas


def test_paragrafo_gigante_sem_quebras_tambem_e_dividido():
    texto = "# Doc\n\n" + ("palavra " * 500)
    trechos = c.segmentar("a/b.md", texto, BASE, limite=300)
    assert all(len(t.conteudo) <= 300 for t in trechos)
    assert len(trechos) > 5


def test_imagens_e_comentarios_html_sao_limpos():
    texto = "# Doc\n\n<!-- nota interna -->\nVeja ![diagrama do fluxo](img/x.png) abaixo."
    conteudo = c.segmentar("a.md", texto, BASE)[0].conteudo
    assert "nota interna" not in conteudo
    assert "[imagem: diagrama do fluxo]" in conteudo
    assert "img/x.png" not in conteudo

from app.business import regras
from app.business.schemas import Trecho, TrechoRecuperado


def _rec(fonte, secao, sim):
    t = Trecho(fonte=fonte, titulo="T", secao=secao, url=f"http://x/{fonte}", conteudo=f"texto de {secao}")
    return TrechoRecuperado(trecho=t, similaridade=sim)


def test_consulta_curta_herda_pergunta_anterior():
    assert regras.montar_consulta("e no ESP32?", ["Como piscar um LED?"]) == "Como piscar um LED? e no ESP32?"


def test_consulta_longa_ou_sem_historico_fica_igual():
    assert regras.montar_consulta("  O que é Node-RED e para que serve?  ", ["x"]) == "O que é Node-RED e para que serve?"
    assert regras.montar_consulta("e no ESP32?", []) == "e no ESP32?"


def test_contexto_suficiente_depende_do_limiar():
    assert regras.tem_contexto_suficiente([_rec("a.md", "s", 0.7)], 0.4)
    assert not regras.tem_contexto_suficiente([_rec("a.md", "s", 0.2)], 0.4)
    assert not regras.tem_contexto_suficiente([], 0.0)


def test_montar_contexto_no_formato_do_lab4():
    contexto = regras.montar_contexto([_rec("a/lab4.md", "Lab 4 > RAG", 0.9), _rec("b.md", "Outro", 0.8)])
    assert "FONTE: [a/lab4.md]\nTÍTULO: Lab 4 > RAG\nCONTEÚDO: texto de Lab 4 > RAG" in contexto
    assert "\n\n---\n\n" in contexto


def test_extrair_citacoes_sem_repeticao():
    texto = "RAG usa embeddings [aulas/lab4.md]. Também [aulas/lab4.md] e [nb/x.ipynb]. Isto [1] não conta."
    assert regras.extrair_citacoes(texto) == ["aulas/lab4.md", "nb/x.ipynb"]


def test_classificar_citacoes_separa_validas_e_inventadas():
    trechos = [_rec("aulas/lab4.md", "s", 0.9)]
    validas, invalidas = regras.classificar_citacoes("a [aulas/lab4.md] b [inventada.md]", trechos)
    assert validas == ["aulas/lab4.md"]
    assert invalidas == ["inventada.md"]


def test_montar_fontes_marca_citadas_e_evita_duplicata():
    trechos = [_rec("a.md", "A > 1", 0.9), _rec("a.md", "A > 1", 0.8), _rec("b.md", "B", 0.7)]
    fontes = regras.montar_fontes(trechos, ["a.md"])
    assert [(f.fonte, f.citada) for f in fontes] == [("a.md", True), ("b.md", False)]

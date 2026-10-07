import zipfile

import pytest

from app import config, ingestao
from tests.conftest import EmbedderFalso


def test_ingestao_de_pasta_local_md_e_ipynb(banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    emb = EmbedderFalso()
    resumo = ingestao.executar(emb, progresso=lambda *_: None)

    assert resumo["arquivos"] == 3                      # .png ignorado
    assert resumo["novos"] == resumo["trechos"] == banco.contar_trechos() > 3
    fontes = {i.trecho.fonte for i in banco.carregar_indice()}
    assert fontes == {"aulas/genAI/lab4/lab4.md", "aulas/genAI/lab2/lab2.md", "aulas/checkpoint/cp4.ipynb"}
    assert banco.obter_metadado("indexado_em")


def test_segunda_ingestao_nao_reembute_e_remove_o_que_sumiu(banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    emb = EmbedderFalso()
    ingestao.executar(emb, progresso=lambda *_: None)
    chamadas = emb.chamadas_documento

    resumo = ingestao.executar(emb, progresso=lambda *_: None)
    assert resumo["novos"] == 0 and emb.chamadas_documento == chamadas   # nada novo, nada embutido

    (material / "aulas" / "genAI" / "lab2" / "lab2.md").unlink()
    resumo = ingestao.executar(emb, progresso=lambda *_: None)
    assert resumo["removidos"] >= 1
    assert "aulas/genAI/lab2/lab2.md" not in {i.trecho.fonte for i in banco.carregar_indice()}


def test_conteudo_alterado_gera_trecho_novo(banco, material, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", str(material))
    emb = EmbedderFalso()
    ingestao.executar(emb, progresso=lambda *_: None)
    arquivo = material / "aulas" / "genAI" / "lab2" / "lab2.md"
    arquivo.write_text(arquivo.read_text(encoding="utf-8") + "\nNovo parágrafo sobre sessões.\n", encoding="utf-8")
    resumo = ingestao.executar(emb, progresso=lambda *_: None)
    assert resumo["novos"] >= 1 and resumo["removidos"] >= 1


def test_download_do_zip_do_repositorio(banco, material, tmp_path, monkeypatch):
    """Simula o zip do GitHub: uma pasta raiz 'Repo-master/' com 'material/' dentro."""
    zip_path = tmp_path / "repo.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for arquivo in material.rglob("*"):
            if arquivo.is_file():
                z.write(arquivo, f"Repo-master/material/{arquivo.relative_to(material).as_posix()}")
    monkeypatch.setattr(config, "MATERIAL_DIR", "")
    monkeypatch.setattr(config, "MATERIAL_ZIP_URL", zip_path.as_uri())

    resumo = ingestao.executar(EmbedderFalso(), progresso=lambda *_: None)
    assert resumo["arquivos"] == 3


def test_material_vazio_levanta_erro(banco, tmp_path, monkeypatch):
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    monkeypatch.setattr(config, "MATERIAL_DIR", str(vazio))
    with pytest.raises(RuntimeError):
        ingestao.executar(EmbedderFalso(), progresso=lambda *_: None)


def test_pasta_inexistente(banco, monkeypatch):
    monkeypatch.setattr(config, "MATERIAL_DIR", "/nao/existe")
    with pytest.raises(FileNotFoundError):
        ingestao.executar(EmbedderFalso(), progresso=lambda *_: None)
